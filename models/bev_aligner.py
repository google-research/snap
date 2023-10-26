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

"""Estimate the relative pose between a pair of overlapping scenes."""
from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections

import snap.configs.defaults as default_configs
from snap.models import base
from snap.models import bev_matcher
from snap.models import layers
from snap.models import pose_estimation
from snap.utils import geometry


class BEVAligner(bev_matcher.BEVMatcher):
  """Estimate the relative pose between a pair of overlapping scenes."""

  def __call__(
      self, data: base.Batch, train: bool = False, debug: bool = False
  ) -> base.Predictions:
    # Compute the BEV features for each scene.
    pred = self.scene_features(data, train, debug)
    plane_i = pred['scene_i']['bev_matching']
    plane_j = pred['scene_j']['bev_matching']
    grid = self.grid.bev()
    batch_size = len(plane_i.features)
    rng_points, rng_poses = jax.random.split(self.make_rng('sampling'), 2)
    rng_points = jax.random.split(rng_points, batch_size)
    rng_poses = jax.random.split(rng_poses, batch_size)

    # Sample query points
    conf_p = pred['scene_i'].get('bev_confidence')
    match self.config.query_point_selection:
      case 'all':
        i_uv_p = grid.grid_index().reshape(-1, 2)
        i_xy_p = (i_uv_p * grid.cell_size)[None].repeat(batch_size, axis=0)
        valid_points = plane_i.valid.reshape(batch_size, -1)
        f_p_i = plane_i.features
        f_p_i = f_p_i.reshape(batch_size, -1, f_p_i.shape[-1])
        if self.config.add_confidence_query:
          conf_p = conf_p.reshape(batch_size, -1)
      case 'random':
        f_p_i, valid_points, i_xy_p, i_uv_p = (
            pose_estimation.sample_sparse_query_points_batched(
                plane_i.features,
                plane_i.valid,
                grid,
                self.config.num_query_points,
                rng_points,
            )
        )
        if self.config.add_confidence_query:
          conf_p = jax.vmap(lambda i, c: c[tuple(i.T)])(i_uv_p, conf_p)
      case 'confidence':
        raise NotImplementedError()
      case _:
        raise ValueError(self.config.query_point_selection)

    # Compute the point-wise scores
    sim_points = jnp.einsum('...nd,...ijd->...nij', f_p_i, plane_j.features)
    if self.config.clip_negative_scores:
      sim_points = jax.nn.relu(sim_points)
    sim_points = sim_points.astype(jnp.float32)
    if self.config.add_temperature:
      sim_points *= jnp.exp(self.temperature)
    prob_points = jax.nn.softmax(sim_points, axis=(-1, -2))

    if self.config.add_confidence_query:
      weights = layers.masked_softmax(conf_p, valid_points, -1)[..., None, None]
      prob_points *= weights
      sim_points *= weights
    else:
      num_valid = valid_points.sum(-1).clip(min=1)[:, None, None, None]
      sim_points /= num_valid
      prob_points /= num_valid

    # Sample poses
    match self.config.pose_selection:
      case 'random':
        j_t_i = pose_estimation.sample_transforms_random_batched(
            rng_poses, self.config.num_pose_samples, grid
        )
      case 'ransac':
        j_t_i = pose_estimation.sample_transforms_ransac_batched(
            rng_poses,
            jax.lax.stop_gradient(prob_points),
            i_xy_p,
            self.config.num_pose_samples,
            self.config.num_pose_sampling_retries,
            grid,
        )
      case _:
        raise ValueError(self.config.pose_selection)
    if 'T_j2i' in data:
      i_t_j_gt = geometry.Transform2D.from_Transform3D(data['T_j2i'])
      j_t_i = jax.tree_util.tree_map(
          lambda *x: jnp.concatenate(x, 1), i_t_j_gt.inv[..., None], j_t_i
      )
    pred['j_t_i_samples'] = j_t_i

    scores_poses = pose_estimation.pose_scoring_many_batched(
        j_t_i,
        sim_points,
        i_xy_p,
        valid_points,
        plane_j.valid,
        grid,
        self.config.mask_score_out_of_bounds,
    )
    pred['scores_poses'] = jax.nn.log_softmax(scores_poses, axis=-1)
    return pred

  @classmethod
  @property
  def default_config(cls) -> ml_collections.ConfigDict:
    return default_configs.bev_aligner()


class BEVAlignerModel(base.BaseModel):
  """Trainer-facing wrapper for the BEVAligner."""

  def build_flax_model(self) -> nn.Module:
    return BEVAligner(self.config, self.dataset_meta_data['grid'], self.dtype)

  @classmethod
  def default_flax_model_config(cls) -> ml_collections.ConfigDict:
    return BEVAligner.default_config  # pytype: disable=bad-return-type

  def loss_metrics_function(
      self,
      pred: base.Predictions,
      data: base.Batch,
      model_params: Optional[jnp.ndarray] = None,
  ) -> base.LossMetricsTuple:
    nll = -pred['scores_poses'][..., 0]
    losses = {'localization/nll': nll, 'total': nll}

    # Discard the GT pose when computing the metrics.
    best_index = jnp.argmax(pred['scores_poses'][..., 1:], axis=-1) + 1
    j_t_i_best = jax.vmap(lambda t, i: t[i])(pred['j_t_i_samples'], best_index)
    i_t_j_gt = geometry.Transform2D.from_Transform3D(data['T_j2i'])
    dr, dt = (j_t_i_best @ i_t_j_gt).magnitude()
    metrics = {
        'loc/err_max_position': dt,
        'loc/err_max_rotation': dr,
        'loc/recall_top1': jnp.argmax(pred['scores_poses'], axis=-1) == 0,
    }
    for t in [0.5, 1, 2]:
      metrics[f'loc/recall_max_{t}m'] = dt < t
      metrics[f'loc/recall_max_{t}°'] = dr < t
    if self.config.add_temperature and model_params is not None:
      metrics['loc/temperature'] = model_params['temperature'].repeat(len(nll))
    return losses, metrics
