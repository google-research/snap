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

from absl import logging
from etils.array_types import FloatArray
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections

import snap.configs.defaults as default_configs
from snap.data import types as data_types
from snap.models import base
from snap.models import bev_mapper
from snap.models import layers
from snap.models import pose_estimation
from snap.models import types
from snap.utils import geometry
from snap.utils import grids


def build_query_frustum_grid(
    cell_size: float,
    depth: float,
    filter_points_in_fov: bool = False,
    hfov_deg: None | float = None,
) -> tuple[grids.Grid2D, FloatArray['3'], FloatArray['N 3']]:
  """Define a gravity-aligned grid bounding the query frustum."""
  width = 3 * depth // 2  # Coarse approximation of SV 72deg HFoV.
  grid = grids.Grid2D.from_extent_meters((width, depth), cell_size)
  grid_p_view = jnp.array([width / 2, 0.0])

  # qgrid is the coordinate frame of the frustum grid
  # while q is the frame of the gravity-aligned scene.
  qgrid_xy_p = grid.index_to_xyz(grid.grid_index())
  q_xy_p = qgrid_xy_p - grid_p_view
  if filter_points_in_fov:
    angle = jnp.arctan2(q_xy_p[..., 0], q_xy_p[..., 1])
    max_angle = hfov_deg / 2
    q_xy_p = q_xy_p[jnp.abs(angle) < jnp.deg2rad(max_angle)][:, None]
  return grid, grid_p_view, q_xy_p


class BEVLocalizer(nn.Module):
  """Estimate the relative pose between a pair of overlapping scenes."""

  config: ml_collections.ConfigDict
  scene_config: data_types.SceneConfig
  grid_map: grids.Grid2D
  semantic_map_classes: None | tuple[str, ...] = None
  dtype: jnp.dtype = jnp.float32

  def __post_init__(self):
    with jax.ensure_compile_time_eval():
      self.grid_query, self.qgrid_p_q, self.q_xy_p = build_query_frustum_grid(
          self.grid_map.cell_size,
          self.config.query_frustum_depth,
          self.config.filter_points_in_fov,
          self.scene_config.streetview_hfov_deg,
      )
    if self.config.get('query_grid_z_offset') is not None:
      logging.warning(
          'query_grid_z_offset is deprecated, adjust'
          ' bev_mapper.scene_z_offset instead.'
      )
    if self.config.get('query_grid_z_range') is not None:
      logging.warning(
          'query_grid_z_range is deprecated, adjust'
          ' bev_mapper.scene_z_offset_range instead.'
      )
    super().__post_init__()

  def setup(self):
    if self.config.add_confidence_map:
      raise NotImplementedError('Map confidence is not yet supported.')
    if self.config.add_confidence_query or self.config.add_confidence_map:
      self.config.bev_mapper.add_confidence = True
    self.bev_mapper = bev_mapper.BEVMapper(
        self.config.bev_mapper,
        self.grid_map,
        self.semantic_map_classes,
        self.dtype,
    )
    self.bev_mapper_query = None
    if self.config.bev_mapper_query is not None:
      self.bev_mapper_query = bev_mapper.BEVMapper(
          self.config.bev_mapper_query,
          self.grid_map,
          self.semantic_map_classes,
          self.dtype,
      )
    if self.config.add_temperature:
      init_temperature = nn.initializers.constant(self.config.init_temperature)
      self.temperature = self.param('temperature', init_temperature, ())

  def recover_dense_feature_plane(
      self, plane_sparse: types.FeaturePlane
  ) -> types.FeaturePlane:
    """Convert a map defined only on valid points in the BEV to a dense map."""
    plane = types.FeaturePlane(
        features=jnp.zeros(
            (*self.grid_query.extent, plane_sparse.features.shape[-1])
        ),
        valid=jnp.zeros(self.grid_query.extent, bool),
    )
    q_xy_p = self.q_xy_p.squeeze(1)
    indices = self.grid_query.xyz_to_index(q_xy_p + self.qgrid_p_q[:2])
    plane.valid = plane.valid.at[tuple(indices.T)].set(
        plane_sparse.valid.reshape(len(indices))
    )
    plane.features = plane.features.at[tuple(indices.T)].set(
        plane_sparse.features.reshape(len(indices), -1)
    )
    return plane

  def __call__(
      self, data: base.Batch, train: bool = False, debug: bool = False
  ) -> base.Predictions:
    batch_size = len(data['query']['images'])
    q_xy_p = self.q_xy_p[None].repeat(batch_size, axis=0)

    pred = {}
    pred['map'] = self.bev_mapper(data['map'], train, debug)
    pred['query'] = (self.bev_mapper_query or self.bev_mapper)(
        data['query'] | dict(xy_bev=q_xy_p),
        train,
        debug,
        is_query=True,
    )

    plane_map = pred['map']['bev_matching']
    plane_q = pred['query']['bev_matching']
    batch_size = len(plane_map.features)
    rng_poses = jax.random.split(self.make_rng('sampling'), batch_size)

    # Gather query points
    q_xy_p = q_xy_p.squeeze(2)  # Remove the dummy spatial dimension.
    valid_points = plane_q.valid.reshape(batch_size, -1)
    f_p_q = plane_q.features
    f_p_q = f_p_q.reshape(batch_size, -1, f_p_q.shape[-1])

    # Compute the point-wise scores
    sim_points = jnp.einsum('...nd,...ijd->...nij', f_p_q, plane_map.features)
    if self.config.clip_negative_scores:
      sim_points = jax.nn.relu(sim_points)
    sim_points = sim_points.astype(jnp.float32)
    if self.config.add_temperature:
      sim_points *= jnp.exp(self.temperature)
    prob_points = jax.nn.softmax(sim_points, axis=(-1, -2))

    if self.config.add_confidence_query:
      conf_p = pred['query']['bev_confidence'].reshape(batch_size, -1)
      weights = layers.masked_softmax(conf_p, valid_points, -1)[..., None, None]
      prob_points *= weights
      sim_points *= weights
    else:
      num_valid = valid_points.sum(-1).clip(min=1)[:, None, None, None]
      sim_points /= num_valid
      prob_points /= num_valid

    # Sample poses
    m_t_q = pose_estimation.sample_transforms_ransac_batched(
        rng_poses,
        jax.lax.stop_gradient(prob_points),
        q_xy_p,
        self.config.num_pose_samples,
        self.config.num_pose_sampling_retries,
        self.grid_map,
    )
    if (m_t_q_gt := data.get('T_query2map')) is not None:
      m_t_q_gt = geometry.Transform2D.from_Transform3D(m_t_q_gt)
      m_t_q = jax.tree_util.tree_map(
          lambda *x: jnp.concatenate(x, 1), m_t_q_gt[..., None], m_t_q
      )
    pred['map_t_query_samples'] = m_t_q

    pred['scores_poses'] = scores = pose_estimation.pose_scoring_many_batched(
        m_t_q,
        sim_points,
        q_xy_p,
        valid_points,
        plane_map.valid,
        self.grid_map,
        self.config.mask_score_out_of_bounds,
    )
    # Discard the GT pose if it was prepended; the loss requires its score only.
    start_idx = int(m_t_q_gt is not None)
    pred['best_index'] = best_idx = jnp.argmax(scores[:, start_idx:], axis=-1)
    fn_batch_indexing = jax.vmap(lambda t, i: t[i])
    pred['map_t_query'] = fn_batch_indexing(m_t_q[:, start_idx:], best_idx)

    if self.config.do_grid_refinement:
      pred['map_t_query_ransac'] = pred['map_t_query']
      pred['map_t_query'], pred['scores_grid_refine'] = (
          pose_estimation.grid_refinement_batched(
              pred['map_t_query'],
              sim_points,
              q_xy_p,
              valid_points,
              plane_map.valid,
              self.grid_map,
              self.config.mask_score_out_of_bounds,
          )
      )

    return pred

  @classmethod
  @property
  def default_config(cls) -> ml_collections.ConfigDict:
    return default_configs.bev_localizer()


class BEVLocalizerModel(base.BaseModel):
  """Trainer-facing wrapper for the BEVAligner."""

  def build_flax_model(self) -> nn.Module:
    return BEVLocalizer(
        self.config,
        self.dataset_meta_data['build_config'].scene_config,
        self.dataset_meta_data['grid'].bev(),
        self.dataset_meta_data['semantic_map_classes'],
        self.dtype,
    )

  @classmethod
  def default_flax_model_config(cls) -> ml_collections.ConfigDict:
    return BEVLocalizer.default_config  # pytype: disable=bad-return-type

  def loss_metrics_function(
      self,
      pred: base.Predictions,
      data: base.Batch,
      model_params: Optional[jnp.ndarray] = None,
  ) -> base.LossMetricsTuple:
    scores = pred['scores_poses']
    m_t_q_gt = geometry.Transform2D.from_Transform3D(data['T_query2map'])
    samples_t_gt = pred['map_t_query_samples'].inv @ m_t_q_gt[..., None]
    dr_samples, dt_samples = samples_t_gt.magnitude()
    if self.config.threshold_remove_accurate_poses is not None:
      dr_min, dt_min = self.config.threshold_remove_accurate_poses
      remove = (dr_samples < dr_min) & (dt_samples < dt_min)
      remove = remove.at[..., 0].set(False)  # Do not remove the GT pose score.
      scores = jnp.where(remove, -jnp.inf, scores)
    nll = -jax.nn.log_softmax(scores, axis=-1)[..., 0]
    losses = {'localization/nll': nll, 'total': nll}

    dr, dt = (pred['map_t_query'].inv @ m_t_q_gt).magnitude()
    metrics = {
        'loc/err_max_position': dt,
        'loc/err_max_rotation': dr,
        'loc/recall_top1': jnp.argmax(pred['scores_poses'], axis=-1) == 0,
    }
    for t in [0.5, 1, 2, 5]:
      metrics[f'loc/recall_max_{t}m'] = dt < t
      metrics[f'loc/recall_max_{t}°'] = dr < t
    if self.config.add_temperature and model_params is not None:
      metrics['loc/temperature'] = model_params['temperature'].repeat(len(nll))
    # Monitor the quality of the samples
    for dt_thresh, dr_thresh in [(0.5, 1), (1, 2), (2, 4)]:
      recall = (dr_samples < dr_thresh) & (dt_samples < dt_thresh)
      recall = jnp.mean(recall[..., 1:], axis=-1)  # remove the GT pose
      metrics[f'loc/recall_samples_{dt_thresh}m_{dr_thresh}°'] = recall
    return losses, metrics
