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

"""Match BEV feature planes in a pair of overlapping scenes."""
from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections

import snap.configs.defaults as default_configs
from snap.models import base
from snap.models import bev_estimator
from snap.models import layers
from snap.models import pose_estimation
from snap.utils import geometry
from snap.utils import grids


class BEVMatcher(nn.Module):
  """Match BEV feature planes in a pair of overlapping scenes."""

  config: ml_collections.ConfigDict
  grid: grids.Grid3D
  dtype: jnp.dtype = jnp.float32

  def setup(self):
    self.config.bev_estimator.update(
        dict(
            matching_dim=self.config.matching_dim,
            normalize_matching_features=self.config.normalize_features,
            add_confidence=self.config.add_confidence_query
            or self.config.add_confidence_ref,
        )
    )
    self.bev_estimator = bev_estimator.BEVEstimator(
        self.config.bev_estimator, self.grid, self.dtype
    )
    if self.config.add_temperature:
      init_temperature = nn.initializers.constant(self.config.init_temperature)
      self.temperature = self.param('temperature', init_temperature, ())

  def scene_features(
      self, data: base.Batch, train: bool = False, debug: bool = False
  ) -> base.Predictions:
    scenes = self.bev_estimator(
        jax.tree_map(
            lambda *t: jnp.concatenate(t), data['scene_i'], data['scene_j']
        )
    )
    scene_i = jax.tree_util.tree_map(lambda x: x[: len(x) // 2], scenes)
    scene_j = jax.tree_util.tree_map(lambda x: x[len(x) // 2 :], scenes)
    return dict(scene_i=scene_i, scene_j=scene_j)

  def __call__(
      self, data: base.Batch, train: bool = False, debug: bool = False
  ) -> base.Predictions:
    # Compute the BEV features for each scene.
    pred = self.scene_features(data, train, debug)
    plane_i = pred['scene_i']['bev_matching']
    plane_j = pred['scene_j']['bev_matching']

    # Sample some sparse query points.
    rng = jax.random.split(self.make_rng('sampling'), len(plane_i.features))
    grid = self.grid.bev()
    f_q_i, valid_q_i, i_xy_q, i_uv_q = (
        pose_estimation.sample_sparse_query_points_batched(
            plane_i.features,
            plane_i.valid,
            rng,
            grid,
            self.config.num_query_points,
        )
    )

    # Match sparse queries in i to dense features in j.
    sim = jnp.einsum('...nd,...xyd->...nxy', f_q_i, plane_j.features)
    sim = sim.astype(jnp.float32)
    if self.config.add_temperature:
      sim = sim * jnp.exp(self.temperature)
    sim = jnp.where(plane_j.valid[:, None], sim, -jnp.inf)
    if self.config.max_distance_negatives is not None:
      # Match only a small neighborhood around the ground truth correspondence.
      i_t_j = geometry.Transform2D.from_Transform3D(data['T_j2i'])
      j_xy_q = i_t_j.inv @ i_xy_q
      j_xy_grid = grid.index_to_xyz(grid.grid_index())
      dist = jnp.abs(j_xy_q[..., None, None, :] - j_xy_grid).max(axis=-1)
      keep = dist < self.config.max_distance_negatives
      sim = jnp.where(keep, sim, -jnp.inf)
    scores = jax.nn.log_softmax(sim, axis=(-1, -2))

    if self.config.add_confidence_query:
      conf_i = pred['scene_i']['bev_confidence']
      pred['conf_q_i'] = jax.vmap(lambda i, c: c[tuple(i.T)])(i_uv_q, conf_i)
    if self.config.add_confidence_ref:
      scores += pred['scene_j']['bev_confidence'][:, None]

    return {
        **pred,
        'matching_scores': scores,
        'valid_q_i': valid_q_i,
        'i_xy_q': i_xy_q,
    }

  @classmethod
  @property
  def default_config(cls) -> ml_collections.ConfigDict:
    return default_configs.bev_matcher()


class BEVMatcherModel(base.BaseModel):
  """Trainer-facing wrapper for the BEVMatcher."""

  def build_flax_model(self) -> nn.Module:
    return BEVMatcher(self.config, self.dataset_meta_data['grid'], self.dtype)

  @classmethod
  def default_flax_model_config(cls) -> ml_collections.ConfigDict:
    return BEVMatcher.default_config  # pytype: disable=bad-return-type

  def loss_metrics_function(
      self,
      pred: base.Predictions,
      data: base.Batch,
      model_params: Optional[jnp.ndarray] = None,
  ) -> base.LossMetricsTuple:
    grid = self.flax_model.grid.bev()
    # Transform the query points from scene i to scene j.
    i_t_j = geometry.Transform2D.from_Transform3D(data['T_j2i'])
    j_xy_q = i_t_j.inv @ pred['i_xy_q']
    j_uv_q = j_xy_q / grid.cell_size

    # Interpolate each score map at the GT location.
    scores = pred['matching_scores']
    scores_gt, valid_q_j = pose_estimation.interpolate_score_maps_batched(
        scores,
        j_uv_q,
        pred['scene_j']['bev_matching'].valid,
    )

    valid = pred['valid_q_i'] & valid_q_j
    if self.config.add_confidence_query:
      # Equivalent to normalize(exp(log_sigmoid)) but more stable.
      weights = layers.masked_softmax(pred['conf_q_i'], valid, -1)
      nll = jnp.sum(scores_gt * weights, axis=-1)
    else:
      nll = -layers.masked_mean(scores_gt, valid, -1)
    losses = {'matching/nll': nll, 'total': nll}

    j_xy_amax = grids.argmax_nd(scores, grid) * grid.cell_size
    err_max = jnp.linalg.norm(j_xy_q - j_xy_amax, axis=-1)
    j_xy_exp = grids.expectation_nd(jnp.exp(scores), grid) * grid.cell_size
    err_exp = jnp.linalg.norm(j_xy_q - j_xy_exp, axis=-1)
    metrics = {
        'matching/err_max': layers.masked_mean(err_max, valid, -1),
        'matching/err_exp': layers.masked_mean(err_exp, valid, -1),
    }
    for t in [0.5, 1, 2]:
      recall_at_t = layers.masked_mean(err_max < t, valid, -1)
      metrics[f'matching/recall_max_{t}m'] = recall_at_t
    if self.config.add_temperature and model_params is not None:
      temperature = model_params['temperature']
      metrics['matching/temperature'] = temperature.repeat(len(scores))
    return losses, metrics
