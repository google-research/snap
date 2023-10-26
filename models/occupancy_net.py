# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Predict per-point occupancy from an image-based scene representation."""
import functools
from typing import Optional

from etils.array_types import BoolArray
from etils.array_types import FloatArray
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections

import snap.configs.defaults as default_configs
from snap.models import base
from snap.models import image_scene_encoder
from snap.models import layers
from snap.models import types
from snap.utils import grids


@functools.partial(jax.jit, static_argnames=['num_samples', 'margin'])
def sample_queries_from_rays(
    hits: FloatArray['N 3'],
    origins: FloatArray['N 3'],
    valid: BoolArray['N'],
    num_samples: int,
    margin: float,
) -> types.LidarRaySamples:
  """Sample num_samples-1 freespace points and add 1 occupied hit point."""
  hits = hits[None]  # Add the sample dimension.
  origins = origins[None]
  direction = hits - origins
  distance = jnp.linalg.norm(direction, axis=2, keepdims=True)
  direction = direction * ((distance - margin) / distance.clip(min=1))
  num_neg = num_samples - 1
  steps = jnp.linspace(0, 1, num_neg)
  samples_neg = steps[:, None, None] * direction + origins
  samples = jnp.concatenate([hits, samples_neg], 0)
  labels = jnp.r_[True, jnp.zeros(num_neg, bool)]
  labels = labels[:, None].repeat(samples.shape[1], axis=1)
  # Flatten samples and points dimensions.
  samples = samples.reshape(-1, 3)
  labels = labels.reshape(-1)
  valid = valid[None].repeat(num_samples, axis=0).reshape(-1)
  return types.LidarRaySamples(points=samples, labels=labels, valid=valid)


sample_queries_from_rays_batched = jax.vmap(
    sample_queries_from_rays, in_axes=(0, 0, 0, None, None)
)


class OccupancyNet(nn.Module):
  """Predict per-point occupancy from an image-based scene representation."""

  config: ml_collections.ConfigDict
  grid: grids.Grid3D
  dtype: jnp.dtype = jnp.float32

  def setup(self):
    self.scene_encoder = image_scene_encoder.ImageSceneEncoder(
        self.config.scene_encoder, self.dtype
    )
    self.mlp_out = layers.MLP(self.config.occupancy_mlp, self.dtype)

  def __call__(
      self, data: base.Batch, train: bool = False, debug: bool = False
  ) -> base.Predictions:
    if 'map' in data:
      data = data['map']
    xyz_grid = self.grid.index_to_xyz(self.grid.grid_index())
    xyz_grid = xyz_grid[None].repeat(len(data['images']), axis=0)  # Add batch.
    pred = self.scene_encoder(data | dict(xyz_query=xyz_grid), train)
    volume = pred['feature_volume']

    queries = data.get('occupancy_queries')
    if queries is None:
      if 'lidar_rays' in data:
        # This could be moved to the dataloader to simplify the logic
        # but would incur additional device transfers.
        rays = data['lidar_rays']
        pred['ray_samples'] = samples = sample_queries_from_rays_batched(
            rays['points'],
            rays['origins'],
            rays['mask'],
            self.config.num_samples_per_ray,
            self.config.ray_margin,
        )
        queries = samples.points
      else:
        raise ValueError('No points or rays given in the data dict.')

    # Simple prediction via trilinear interpolation.
    indices = queries / self.grid.cell_size
    features, valid = jax.vmap(grids.interpolate_nd)(
        volume.features, indices, volume.valid
    )
    logits = self.mlp_out(features, train).squeeze(-1).astype(jnp.float32)
    occupancy = types.OccupancySamples(
        values=nn.sigmoid(logits), valid=valid, logits=logits
    )

    return {
        **pred,
        'occupancy': occupancy,
    }

  @classmethod
  @property
  def default_config(cls) -> ml_collections.ConfigDict:
    return default_configs.occupancy_net()


class OccupancyNetModel(base.BaseModel):
  """Trainer-facing wrapper for the OccupancyNet."""

  def build_flax_model(self) -> nn.Module:
    return OccupancyNet(self.config, self.dataset_meta_data['grid'], self.dtype)

  @classmethod
  def default_flax_model_config(cls) -> ml_collections.ConfigDict:
    return OccupancyNet.default_config  # pytype: disable=bad-return-type

  def loss_metrics_function(
      self,
      pred: base.Predictions,
      data: base.Batch,
      model_params: Optional[jnp.ndarray] = None,
  ) -> base.LossMetricsTuple:
    labels = pred['ray_samples'].labels
    logits = pred['occupancy'].logits
    occ = logits > 0
    # Compute the loss and metrics only on points visible by at least one view.
    mask = pred['occupancy'].valid & pred['ray_samples'].valid

    log_p = jax.nn.log_sigmoid(logits)
    log_not_p = jax.nn.log_sigmoid(-logits)
    bce_per_sample = -jnp.where(labels, log_p, log_not_p)
    bce_pos = layers.masked_mean(bce_per_sample, mask & labels, 1)
    bce_neg = layers.masked_mean(bce_per_sample, mask & (~labels), 1)
    bce = (bce_pos + bce_neg) / 2
    losses = {
        'occupancy_bce': bce,
        'total': bce,
    }

    correct = occ == labels
    metrics = {
        'occupancy/accuracy': layers.masked_mean(correct, mask, 1),
        'occupancy/recall': layers.masked_mean(correct, mask & labels, 1),
        'occupancy/precision': layers.masked_mean(correct, mask & (~labels), 1),
    }

    return losses, metrics
