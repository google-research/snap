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

"""Predict a top-down Bird's-Eye-View feature plane from images."""
import pprint
from typing import Any

from absl import logging
from etils.array_types import BoolArray
from etils.array_types import FloatArray
import flax.linen as nn
from flax.training import checkpoints
import jax
import jax.numpy as jnp
import ml_collections
from scenic.google.xm import xm_utils

import snap.configs.defaults as default_configs
from snap.models import base
from snap.models import image_encoder
from snap.models import streetview_encoder
from snap.models import layers
from snap.models import semantic_raster_encoder
from snap.models import types
from snap.utils import configs as config_utils
from snap.utils import grids
from snap.utils import misc


class VerticalPooling(nn.Module):
  """Flatten a 3D scene volume into a 2D BEV by pooling vertically."""

  config: ml_collections.ConfigDict
  dtype: jnp.dtype = jnp.float32

  pooling_ops = {k: getattr(jnp, k) for k in ('max', 'sum', 'mean')}

  def setup(self):
    if self.config.pooling in ('weighted', 'softmax'):
      self.confidence_head = nn.Dense(1, param_dtype=self.dtype)
    elif self.config.pooling == 'mlp':
      self.fusion_mlp = layers.MLP(self.config.mlp, self.dtype)
    elif self.config.pooling not in self.pooling_ops:
      raise NotImplementedError(self.config.pooling)

  def __call__(self, feature_volume: types.FeatureVolume) -> base.Predictions:
    valid = feature_volume.valid
    valid_any = valid.any(-1)
    # We apply the double-where trick to avoid NaNs in gradients.
    valid_any_or_all = jnp.where(valid_any[..., None], valid, 1)

    pred = {}
    if self.config.pooling in ('weighted', 'softmax'):
      scores = self.confidence_head(feature_volume.features).squeeze(-1)
      scores = scores.astype(jnp.float32)
      if self.config.pooling == 'weighted':
        scores = jax.nn.log_sigmoid(scores)
      pred['scores'] = scores
      weights = jax.nn.softmax(
          scores, where=valid_any_or_all, initial=0, axis=-1
      )
      weights = pred['weights'] = jnp.where(valid, weights, 0)
      features = jnp.sum(feature_volume.features * weights[..., None], -2)
      features = features.astype(feature_volume.features.dtype)
    elif self.config.pooling == 'mlp':
      features = jnp.where(valid[..., None], feature_volume.features, 0)
      features = features.reshape(*features.shape[:-2], -1)
      features = self.fusion_mlp(features)
    else:
      kwargs = dict(axis=-2, where=valid_any_or_all[..., None])
      if self.config.pooling == 'max':
        kwargs['initial'] = -jnp.inf
      features = self.pooling_ops[self.config.pooling](
          feature_volume.features, **kwargs
      )
    features = jnp.where(valid_any[..., None], features, 0)
    pred['plane'] = types.FeaturePlane(features=features, valid=valid_any)
    return pred


class BEVMapper(nn.Module):
  """Encode a set of images into a 2D feature plane."""

  config: ml_collections.ConfigDict
  grid: grids.Grid2D
  semantic_map_classes: None | tuple[str, ...] = None
  dtype: jnp.dtype = jnp.float32

  def __post_init__(self):
    if (xid := self.config.pretrained_xid) is not None:
      pretrained_config, workdir = xm_utils.get_info_from_xmanager(xid, 1)
      pretrained_config = pretrained_config.model.bev_mapper
      diff = config_utils.config_diff(self.config, pretrained_config)
      if diff:
        logging.warning(
            'Found differences between configs:\n%s', pprint.pformat(diff)
        )
      self.config.pretrained_path = workdir
    super().__post_init__()

  def setup(self):
    feature_dimensions = []
    self.streetview_encoder = self.aerial_encoder = self.semantic_encoder = None
    if self.config.streetview_encoder is not None:
      self.streetview_encoder = streetview_encoder.StreetViewEncoder(
          self.config.streetview_encoder, self.dtype
      )
      self.vertical_pooling = VerticalPooling(self.config.pooling, self.dtype)
      feature_dimensions.append(self.config.streetview_encoder.feature_dim)
    if self.config.aerial_encoder is not None:
      self.aerial_encoder = image_encoder.ImageEncoder(
          self.config.aerial_encoder, self.dtype
      )
      feature_dimensions.append(self.config.aerial_encoder.output_dim)
    if self.config.semantic_encoder is not None:
      self.semantic_encoder = semantic_raster_encoder.SemanticRasterEncoder(
          self.config.semantic_encoder, self.semantic_map_classes, self.dtype
      )
      feature_dimensions.append(self.config.semantic_encoder.encoder.output_dim)
    if not feature_dimensions:
      raise ValueError('Need to create at least one input encoder.')
    elif len(feature_dimensions) > 1:
      if not all(d == feature_dimensions[0] for d in feature_dimensions):
        raise ValueError(
            f'Encoder have different output dimensions: {feature_dimensions}'
        )
      self.modality_fusion = VerticalPooling(
          self.config.modality_fusion, self.dtype
      )

    if self.config.bev_net is not None:
      raise NotImplementedError('BEV network not yet implemented')
    if self.config.matching_dim is not None:
      # Initialize the projection such that the dot product has unit variance.
      self.matching_proj = nn.Dense(
          self.config.matching_dim,
          kernel_init=jax.nn.initializers.variance_scaling(
              1 / jnp.sqrt(self.config.matching_dim),
              'fan_in',
              'truncated_normal',
          ),
          param_dtype=self.dtype,
      )
    if self.config.add_confidence:
      self.confidence_head = nn.Sequential(
          [nn.Dense(1, param_dtype=self.dtype)]
      )

  def encode_streetview(
      self, data: base.Batch, train: bool, is_query: bool
  ) -> base.Predictions:
    if 'xyz_query' not in data:
      scene_t_view = data['T_view2scene']
      xy = data.get('xy_bev')
      if xy is None:
        xy = self.grid.index_to_xyz(self.grid.grid_index())
      if len(xy.shape) != 4:  # Add batch dimension.
        xy = xy[None].repeat(len(scene_t_view), axis=0)
      if (z_offset := data.get('z_offset')) is None:
        # Horizontal plane at a fixed height w.r.t the cameras
        camera_heights = jnp.median(scene_t_view.t[..., -1], axis=-1)
        # Backward compatibility with data pipeline (4m).
        height_below_camera = self.config.get('scene_z_offset', 4.0)
        z_offset = camera_heights - height_below_camera
        if (
            train
            and is_query
            and self.config.get('scene_z_offset_range') is not None
        ):
          z_min, z_max = self.config.get('scene_z_offset_range')
          z_offset = z_offset + jax.random.uniform(
              self.make_rng('sampling'),
              z_offset.shape,
              minval=z_min,
              maxval=z_max,
          )
      scene_z_height = self.config.get('scene_z_height', 12.0)
      z = (
          jnp.arange(0, scene_z_height, self.grid.cell_size)[None]
          + z_offset[:, None]
          + self.grid.cell_size / 2  # To voxel centers.
      )
      xy, z = jnp.broadcast_arrays(
          xy[:, :, :, None, :], z[:, None, None, :, None]
      )
      data['xyz_query'] = jnp.concatenate([xy, z[..., :1]], axis=-1)

    pred = self.streetview_encoder(data, train=train)
    pred['vertical_pooling'] = self.vertical_pooling(pred['feature_volume'])
    pred['feature_plane'] = pred['vertical_pooling'].pop('plane')
    return pred

  def encode_aerial(
      self, aerial_rgb: FloatArray['B H W 3'], train: bool = False
  ) -> base.Predictions:
    aerial_pyramid = self.aerial_encoder(aerial_rgb, train=train)
    aerial_features = aerial_pyramid.features[-1]  # highest-res features
    aerial_plane = types.FeaturePlane(
        features=aerial_features,
        valid=jnp.ones(aerial_features.shape[:-1], dtype=bool),
    )
    return {'feature_plane': aerial_plane}

  def encode_semantics(
      self, semantic_raster: BoolArray['B H W N'], train: bool = False
  ) -> base.Predictions:
    semantic_pyramid = self.semantic_encoder(semantic_raster, train=train)
    semantic_features = semantic_pyramid.features[-1]  # highest-res features
    semantic_plane = types.FeaturePlane(
        features=semantic_features,
        valid=jnp.ones(semantic_features.shape[:-1], dtype=bool),
    )
    return {'feature_plane': semantic_plane}

  def fuse_neural_maps(
      self, planes: list[types.FeaturePlane], train: bool = False
  ) -> types.FeaturePlane:
    if not planes:
      raise ValueError('No feature plane given.')
    elif len(planes) == 1:
      return planes[0]

    # Fuse the feature planes inferred from the different inputs.
    if train and self.config.apply_modality_dropout:
      dropout_mask = jax.random.bernoulli(
          self.make_rng('sampling'),
          # num modalities x batch
          shape=(len(planes), len(planes[0].features)),
      )
      # If all modalities are masked, we retain all to maximize supervision.
      dropout_mask = jnp.where(
          jnp.any(dropout_mask, axis=0, keepdims=True), dropout_mask, True
      )
      planes = [
          p.replace(valid=jnp.where(m[..., None, None], p.valid, False))
          for p, m in zip(planes, dropout_mask)
      ]
    planes_stacked = types.FeatureVolume(
        features=jnp.stack([f.features for f in planes], axis=-2),
        valid=jnp.stack([f.valid for f in planes], axis=-1),
    )
    return self.modality_fusion(planes_stacked)['plane']

  def __call__(
      self,
      data: base.Batch,
      train: bool = False,
      debug: bool = False,
      is_query: bool = False,
  ) -> base.Predictions:
    pred = {}
    # Run inference for each modality.
    feature_planes = []
    if self.streetview_encoder is not None:
      pred['streetview'] = self.encode_streetview(
          data, train=train, is_query=is_query
      )
      feature_planes.append(pred['streetview']['feature_plane'])
    if self.aerial_encoder is not None and 'rasters' in data:
      # There is no aerial data for query images.
      pred['aerial'] = self.encode_aerial(data['rasters']['rgb'], train=train)
      feature_planes.append(pred['aerial']['feature_plane'])
    if self.semantic_encoder is not None and 'rasters' in data:
      # There are no semantic rasters for query images.
      pred['semantic'] = self.encode_semantics(
          data['rasters']['semantics'], train=train
      )
      feature_planes.append(pred['semantic']['feature_plane'])
    if not feature_planes:
      raise ValueError('No map encoder given.')
    pred['bev_features'] = plane = self.fuse_neural_maps(feature_planes)

    # Compute auxiliary outputs.
    if self.config.matching_dim is not None:
      f_matching = self.matching_proj(plane.features)
      if self.config.normalize_matching_features:
        f_matching = layers.normalize(f_matching)
      f_matching = jnp.where(plane.valid[..., None], f_matching, 0)
      pred['bev_matching'] = types.FeaturePlane(
          features=f_matching, valid=plane.valid
      )
    if self.config.add_confidence:
      scores = self.confidence_head(plane.features).squeeze(-1)
      conf = nn.log_sigmoid(scores.astype(jnp.float32))
      pred['bev_confidence'] = jnp.where(plane.valid, conf, 0)
    return pred

  @classmethod
  @property
  def default_config(cls) -> ml_collections.ConfigDict:
    return default_configs.bev_mapper()

  def load_pretrained_variables(self) -> None | dict[str, Any]:
    if (path := self.config.pretrained_path) is None:
      return
    state = checkpoints.restore_checkpoint(path, None)
    params = misc.find_nested_dict(state['params'], 'bev_mapper')
    if params is None:
      raise ValueError(f'No parameters for {self.__class__.__name__} in {path}')
    logging.info(
        'Loaded pretrained weights for %s from %s.',
        self.__class__.__name__,
        path,
    )
    return {'params': params}
