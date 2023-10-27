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

"""Encode a set of images into a 3D feature grid."""
import copy
import functools
import itertools
from typing import Any, Optional, Tuple

from absl import logging
from etils.array_types import BoolArray
from etils.array_types import FloatArray
from etils.array_types import IntArray
import flax.linen as nn
from flax.training import checkpoints
import jax
import jax.numpy as jnp
import ml_collections
from scenic.google.xm import xm_utils

import snap.configs.defaults as default_configs
from snap.models import base
from snap.models import image_encoder
from snap.models import layers
from snap.models import types
from snap.utils import configs as config_utils
from snap.utils import geometry
from snap.utils import grids
from snap.utils import misc


@functools.partial(jax.vmap, in_axes=(0, 0, None), out_axes=1)  # views
def project_points_to_views(
    scene_t_view: geometry.Transform3D,
    camera: geometry.Camera,
    points: FloatArray['N 3'],
) -> Tuple[
    FloatArray['N D'],
    BoolArray['N'],
    FloatArray['N'],
    FloatArray['N 3'],
]:
  """Project a set of points to a view and sample features."""
  points_view = scene_t_view.inv @ points
  depth = points_view[..., -1]
  distance = jnp.linalg.norm(points_view, axis=-1, keepdims=True)
  rays = points_view / distance.clip(min=1e-5)
  p2d, vis = camera.world2image(points_view)
  p2d = jnp.flip(p2d, axis=-1)  # xy to ij indexing
  return p2d, vis, depth, rays


project_points_to_views_batched = jax.vmap(
    project_points_to_views, in_axes=(0, 0, 0)
)
gather_batched_observations = jax.vmap(jax.vmap(lambda x, i: x[i]))


@functools.partial(jax.vmap, in_axes=0)  # batch
@functools.partial(jax.vmap, in_axes=0, out_axes=1)  # views
def interpolate_views_all(
    array: FloatArray['H W D'], points: FloatArray['N 2']
) -> FloatArray['N D']:
  """Interpolation of all observations in all views."""
  interp, _ = grids.interpolate_nd(array, points)
  return interp


# Map over all parallel dimensions.
@functools.partial(jax.vmap, in_axes=0)  # batch
@functools.partial(jax.vmap, in_axes=(None, 0, 0))  # points
@functools.partial(jax.vmap, in_axes=(None, 0, 0))  # observations
@functools.partial(jax.vmap, in_axes=(-1, None, None), out_axes=-1)  # channels
def interpolate_views_selective(
    arrays: FloatArray['V H W'], point: FloatArray['2'], index: IntArray
) -> FloatArray:
  """Efficient bilinear interpolation for a subset of views per point."""
  point = point.astype(arrays.dtype)
  size = jnp.asarray(arrays.shape[-2:])
  # The origin of the input is the corner of element (0, 0)
  # but the origin of the indexing is its center.
  # Points that are between the center and the boundary are clipped.
  point = jnp.maximum(jnp.minimum(point - 0.5, size - 1), 0)
  lower = jnp.floor(point).astype(jnp.int32)
  upper = lower + 1
  w_upper = point - lower
  w_lower = 1.0 - w_upper

  weights = [w_lower, w_upper]
  coords = [lower, upper]
  values = []
  for i, j in itertools.product(range(2), repeat=2):
    w = weights[i][0] * weights[j][1]
    values.append(w * arrays[index, coords[i][0], coords[j][1]])
  return sum(values)


# Map over batch, points, views.
@functools.partial(jax.vmap, in_axes=(0, 0, None))
@functools.partial(jax.vmap, in_axes=(0, 0, None))
@functools.partial(jax.vmap, in_axes=(0, 0, None))
def interpolate_depth_score(
    score_scales: FloatArray['D'],
    depth: FloatArray,
    depth_min_max: Tuple[float, float],
) -> FloatArray:
  """Interpolate a 1D depth distribution at point reprojections."""
  num_bins = score_scales.shape[-1]
  min_, max_ = depth_min_max
  depth = depth.clip(min_, max_)
  t = jnp.log(depth / min_) / jnp.log(max_ / min_)
  index = 0.5 + t * (num_bins - 1)  # Map [0, 1] to [0.5, num_bins-0.5].
  score_point, _ = grids.interpolate_nd(score_scales[:, None], index[None])
  return score_point.squeeze(-1)


def view_selection(
    points: FloatArray['... N 3'],
    scene_t_view: geometry.Transform3D,
    vis: FloatArray['... N V'],
    num: int,
) -> Tuple[IntArray['... N K'], FloatArray['... N']]:
  diff = points[..., None, :] - scene_t_view.t[..., None, :, :]  # B,N,V,3
  dist = jnp.linalg.norm(diff, axis=-1)
  dist = jnp.where(vis, dist, jnp.inf)
  min_dist = jnp.min(dist, axis=-1)
  _, indices = jax.lax.top_k(-dist, k=num)  # B,N,K
  return indices, min_dist


@functools.partial(jax.checkpoint, static_argnums=(3, 4))
def pool_multiview_features(
    feats: FloatArray['... V D'],
    valid: BoolArray['... V'],
    scores: Optional[FloatArray['... V']] = None,
    add_minmax: bool = True,
    use_variance: bool = True,
) -> Tuple[FloatArray['... C'], BoolArray['...']]:
  """Extract statistics from multi-view features observing the same point."""
  valid_any = valid.any(-1)
  # We apply the double-where trick to avoid NaNs in gradients.
  valid_ = jnp.where(valid_any[..., None], valid, 1)[..., None]
  if scores is None:
    mean_ = jnp.mean(feats, axis=-2, where=valid_)
    var_ = jnp.var(feats, axis=-2, where=valid_)
  else:
    weights = jax.nn.softmax(
        scores.astype(jnp.float32)[..., None], axis=-2, where=valid_, initial=0
    )
    weights = jnp.where(valid_, weights, 0)
    mean_ = jnp.sum(weights * feats, axis=-2)
    var_ = jnp.sum(weights * (feats - mean_[..., None, :]) ** 2, axis=-2)
    mean_ = mean_.astype(feats.dtype)
    var_ = var_.astype(feats.dtype)
  stats = [mean_]
  if use_variance:
    stats.append(var_)
  if add_minmax:
    max_ = jnp.max(feats, axis=-2, where=valid_, initial=-jnp.inf)
    min_ = jnp.min(feats, axis=-2, where=valid_, initial=jnp.inf)
    stats.extend([max_, min_])
  if scores is not None:
    score_max = jnp.max(
        scores[..., None], axis=-2, where=valid_, initial=-jnp.inf
    )
    stats.append(score_max)
  stats = jnp.where(valid_any[..., None], jnp.concatenate(stats, -1), 0)
  return stats, valid_any


class StreetViewEncoder(nn.Module):
  """Encode a set of images into a 3D feature grid."""

  config: ml_collections.ConfigDict
  dtype: jnp.dtype = jnp.float32

  def __post_init__(self):
    if (xid := self.config.pretrained_xid) is not None:
      pretrained_config, workdir = xm_utils.get_info_from_xmanager(xid, 1)
      self.config = config_utils.configs_merge(
          self.config,
          pretrained_config.model.bev_mapper.streetview_encoder,
      )
      self.config.pretrained_path = workdir
    super().__post_init__()

  def setup(self):
    # Extract features from multiple views in multiple scenes.
    self.image_encoder = nn.vmap(
        image_encoder.ImageEncoder,
        in_axes=(0, None),  # batch the input dict but not train flag
        out_axes=0,
        variable_axes={'params': None, 'batch_stats': None},
        split_rngs={'params': False, 'batch_stats': False},
    )(self.config.image_encoder, self.dtype)
    self.fusion_mlp = nn.remat(layers.MLP)(self.config.fusion, self.dtype)

    if self.config.do_weighted_fusion:
      # Project fusion features and depth scores within a single linear layer.
      proj_config = copy.deepcopy(self.config.proj_mlp)
      proj_config.layers = (
          self.config.feature_dim + self.config.num_scale_bins,
      )
      self.proj_mlp = layers.MLP(proj_config, self.dtype)
    elif self.config.depth_mlp is not None:
      self.depth_mlp = nn.remat(layers.MLP)(self.config.depth_mlp, self.dtype)

  def __call__(self, data: base.Batch, train: bool = False) -> base.Predictions:
    if (f_image_pyr := data.get('image_feature_pyr')) is None:
      images = data['images'].astype(self.dtype)
      f_image_pyr = self.image_encoder(images, train)

    f_images = f_image_pyr.features[-1]  # Get the highest resolution features.
    feature_stride = f_image_pyr.strides[-1][0]  # remove vmapped axis
    cameras = data['camera'].scale(1 / feature_stride[::-1])  # (i,j) to (x,y)
    scene_t_view = data['T_view2scene']
    pred = {'image_feature_pyramid': f_image_pyr}

    if self.config.do_weighted_fusion:
      f_images = self.proj_mlp(f_images, train)
      pred['scores_images'] = f_images[..., -self.config.num_scale_bins :]

    # Compute the locations of 2D observations in all views for all points.
    xyz = data['xyz_query']
    grid_shape = xyz.shape[:-1]  # B,X,Y,Z or B,XY,Z
    xyz_flat = xyz.reshape(len(xyz), -1, 3)
    p2d_views, visible, depth, rays = project_points_to_views_batched(
        scene_t_view, cameras, xyz_flat
    )

    # Select a subset of views for each point and gather their observations.
    if (k_vs := self.config.top_k_view_selection) and f_images.shape[1] > k_vs:
      view_indices, min_distance = view_selection(
          xyz_flat, scene_t_view, visible, k_vs
      )
      p2d_views, visible, depth, rays = (
          gather_batched_observations(x, view_indices)
          for x in (p2d_views, visible, depth, rays)
      )
      f_proj = interpolate_views_selective(f_images, p2d_views, view_indices)
    else:
      f_proj = interpolate_views_all(f_images, p2d_views.swapaxes(1, 2))  # V,N
      min_distance = None

    if self.config.do_weighted_fusion:
      f_proj, scores_scales = jnp.split(
          f_proj, [self.config.feature_dim], axis=-1
      )
      scores_proj = interpolate_depth_score(
          scores_scales, depth, self.config.depth_min_max
      )
    else:
      scores_proj = None
      if self.config.depth_mlp is not None:
        log_depth = jnp.log10(depth.clip(min=0.1, max=100))
        rays = jnp.where(visible[..., None], rays, 0)
        f_proj_depth = jnp.concatenate([f_proj, log_depth[..., None], rays], -1)
        f_proj = f_proj + self.depth_mlp(f_proj_depth, train)
    f_pooled, valid = pool_multiview_features(
        f_proj,
        visible,
        scores_proj,
        self.config.fusion_add_minmax,
        self.config.fusion_use_variance,
    )
    if (
        self.config.get('max_view_distance') is not None
        and min_distance is not None
    ):
      valid = valid & (min_distance <= self.config.max_view_distance)

    f_grid = self.fusion_mlp(f_pooled, train)
    f_grid = jnp.where(valid[..., None], f_grid, 0)
    grid_shape = (-1, *xyz.shape[-4:-1])
    f_grid = f_grid.reshape(*grid_shape, f_grid.shape[-1])
    valid = valid.reshape(grid_shape)
    pred['feature_volume'] = types.FeatureVolume(features=f_grid, valid=valid)
    return pred

  @classmethod
  @property
  def default_config(cls) -> ml_collections.ConfigDict:
    return default_configs.streetview_encoder()

  def load_pretrained_variables(self) -> None | dict[str, Any]:
    if (path := self.config.pretrained_path) is None:
      return
    state = checkpoints.restore_checkpoint(path, None)
    params = misc.find_nested_dict(state['params'], 'streetview_encoder')
    if params is None:
      raise ValueError(f'No parameters for {self.__class__.__name__} in {path}')
    logging.info(
        'Loaded pretrained weights for %s from %s.',
        self.__class__.__name__,
        path,
    )
    return {'params': params}
