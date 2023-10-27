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

"""Default configs for modules and datasets."""
import enum
from typing import Any, Iterable
from ml_collections import config_dict

DATA_DIR = 'path/to/data'

DATA_SPLITS_CITIES = dict(
    train=[
        # Europe
        'barcelona',
        'london',
        'paris',
        # NA
        'manhattan',
        'sanfrancisco',
        'brooklyn',
        # Asia
        'manila',
        'singapore',
        'taiwan',
        'tokyo1',
        # SA
        'rio',
        # Oceania
        'sydney',
    ],
)


class MapModalities(str, enum.Enum):
  STREETVIEW = 'streetview'
  AERIAL = 'aerial'
  SEMANTIC = 'semantic'


def parse_argument_string(args_str: None | str) -> dict[str, Any]:
  args = dict(kv.split('=') for kv in (args_str or '').split(',') if kv)
  default_args = dict(
      image_encoder='R50',
      modalities='streetview+aerial',
  )
  if unknown_args := (set(args) - set(default_args)):
    raise ValueError(f'Unknown args: {unknown_args}')
  return default_args | args


def base() -> config_dict.ConfigDict:
  return config_dict.ConfigDict(
      dict(
          model_name=config_dict.placeholder(str),
          model=config_dict.placeholder(config_dict.ConfigDict),
          data=config_dict.placeholder(config_dict.ConfigDict),
          # training
          batch_size=1,
          rng_seed=0,  # or make it random by default?
          shuffle_seed=1234567,
          data_dtype_str='float32',
          dtype_str='float32',
          lr_configs=dict(
              learning_rate_schedule='compound',
              factors='constant',
              base_learning_rate=config_dict.placeholder(float),
              start_decay_step=0,
              steps_per_cycle=config_dict.placeholder(int),
          ),
          optimizer_configs=dict(optimizer='adam'),
          max_grad_norm=config_dict.placeholder(float),
          num_training_steps=config_dict.placeholder(int),
          num_training_epochs=config_dict.placeholder(int),
          checkpoint=True,
          checkpoint_steps=config_dict.placeholder(int),
          # logging
          log_eval_steps=1000,
          write_summary=True,
          log_summary_steps=config_dict.placeholder(int),
          debug_train=False,
          debug_eval=False,
      )
  ).lock()


def streetview_singlescene() -> config_dict.ConfigDict:
  return config_dict.ConfigDict(
      dict(
          name='streetview_singlescene',
          version=config_dict.placeholder(str),
          data_dir=DATA_DIR,
          dirname='lemming_streetview',
          locations=dict(
              training=config_dict.placeholder(str),
              evaluation=config_dict.placeholder(str),
          ),
          training_size_per_builder=config_dict.placeholder(int),
          evaluation_size=1024,
          voxel_size=config_dict.placeholder(float),
          add_images=True,
          add_lidar_rays=False,
          num_rays=config_dict.placeholder(int),
          pair_scenes=config_dict.placeholder(bool),  # deprecated
          mode=config_dict.placeholder(str),
          add_rasters=False,
          pipeline_options=tuple(),
      )
  ).lock()


def mlp() -> config_dict.ConfigDict:
  return config_dict.ConfigDict(
      dict(
          activation='relu',
          layers=config_dict.placeholder(tuple),
          apply_input_activation=False,
      )
  ).lock()


def resnet(name: str = 'R50') -> config_dict.ConfigDict:
  """ResNet encoder."""
  ret = config_dict.ConfigDict(
      dict(
          width=1,
          depth=50,  # 50/101/152, or list of block depths
          limit_num_blocks=4,
          skip_root_block=False,
          checkpoint_blocks=False,
          checkpoint_units=False,
          pretrained_path='path_to/checkpoint.npz',
      )
  ).lock()
  match name:
    case 'R50':
      pass  # default value
    case 'R152x2':
      ret.update(
          dict(
              width=2,
              depth=152,
              limit_num_blocks=3,
              checkpoint_blocks=True,
              checkpoint_units=True,
              pretrained_path='path_to/checkpoint.npz',
          )
      )
    case 'R101':
      ret.update(
          dict(
              depth=101,
              limit_num_blocks=4,
              checkpoint_blocks=True,
              checkpoint_units=True,
              pretrained_path='path_to/checkpoint.npz',
          )
      )
    case _:
      raise ValueError(f'Unknown ResNet name: {name}')
  return ret


def image_encoder() -> config_dict.ConfigDict:
  return config_dict.ConfigDict(
      dict(
          encoder_name='resnet',
          encoder=resnet(),
          output_dim=128,
          num_pyr_levels=config_dict.placeholder(int),
      )
  ).lock()


def aerial_encoder() -> config_dict.ConfigDict:
  encoder = image_encoder()
  encoder.encoder.skip_root_block = True
  return encoder


def semantic_raster_encoder() -> config_dict.ConfigDict:
  encoder = image_encoder()
  encoder.encoder.skip_root_block = True
  encoder.encoder.depth = 26
  encoder.encoder.width = 2
  encoder.encoder.pretrained_path = None
  encoder.encoder.limit_num_blocks = 4
  return config_dict.ConfigDict(dict(encoder=encoder, embedding_dim=8)).lock()


def streetview_encoder() -> config_dict.ConfigDict:
  feature_dim = 128
  fusion = mlp()
  fusion.layers = (feature_dim * 2, feature_dim)
  proj = mlp()
  proj.apply_input_activation = True
  return config_dict.ConfigDict(
      dict(
          image_encoder=image_encoder(),
          feature_dim=feature_dim,
          fusion=fusion,
          proj_mlp=proj,
          depth_mlp=config_dict.placeholder(config_dict.ConfigDict),
          do_weighted_fusion=True,
          num_scale_bins=32,
          top_k_view_selection=4,
          depth_min_max=(1.0, 32.0),
          fusion_add_minmax=False,
          fusion_use_variance=True,
          max_view_distance=config_dict.placeholder(float),
          pretrained_path=config_dict.placeholder(str),
      )
  ).lock()


def vertical_pooling() -> config_dict.ConfigDict:
  feature_dim = 128
  fusion = mlp()
  fusion.layers = (feature_dim * 2, feature_dim)
  return config_dict.ConfigDict(
      dict(
          pooling='max',
          mlp=fusion,  # Only used for pooling = 'mlp'.
      )
  ).lock()


def bev_mapper(
    modalities: Iterable[str] = (MapModalities.STREETVIEW, MapModalities.AERIAL)
) -> config_dict.ConfigDict:
  """Config for the multi-modal BEV mapper."""
  config = config_dict.ConfigDict(
      dict(
          streetview_encoder=config_dict.placeholder(config_dict.ConfigDict),
          scene_z_offset=4.0,
          scene_z_offset_range=(-2,2),
          scene_z_height=12.0,
          pooling=vertical_pooling(),
          aerial_encoder=config_dict.placeholder(config_dict.ConfigDict),
          semantic_encoder=config_dict.placeholder(config_dict.ConfigDict),
          modality_fusion=vertical_pooling(),
          bev_net=config_dict.placeholder(config_dict.ConfigDict),
          matching_dim=32,
          normalize_matching_features=True,
          add_confidence=False,
          apply_modality_dropout=True,
          pretrained_path=config_dict.placeholder(str),
      )
  )
  for m in modalities:
    match m:
      case MapModalities.STREETVIEW:
        config.streetview_encoder = streetview_encoder()
      case MapModalities.AERIAL:
        config.aerial_encoder = aerial_encoder()
      case MapModalities.SEMANTIC:
        config.semantic_encoder = semantic_raster_encoder()
      case _:
        raise ValueError(f'Unknown modality: {m}')
  return config.lock()


def occupancy_net() -> config_dict.ConfigDict:
  predictor = mlp()
  predictor.layers = (128, 1)
  return config_dict.ConfigDict(
      dict(
          num_samples_per_ray=100,
          ray_margin=0.2,
          streetview_encoder=streetview_encoder(),
          occupancy_mlp=predictor,
      )
  ).lock()


def semantic_net() -> config_dict.ConfigDict:
  return config_dict.ConfigDict(
      dict(
          bev_mapper=bev_mapper(),
          decoder_type='mlp',
          decoder_dim=128,
          mlp_num_layers=2,
          resnet_num_units=8,
          apply_random_flip=False,
          area_classes=(
              'crosswalk',
              'sidewalk',
              'road',
              'terrain',
              # 'line',
              # 'stopline',
              # 'otherlanemarking',
              'building',
          ),
          area_frequencies=(
              ('crosswalk', 0.036434),
              ('sidewalk', 0.226553),
              ('road', 0.446990),
              ('terrain', 0.085374),
              ('building', 0.204649),
          ),
          object_classes_exclusive=(
              'fence',
              # 'wall',
              # 'bus_stop',
              # 'phone_booth',
              'pole',
              # 'fire_hydrant',
              # 'guard_rail',
              'tree',
              # 'line',
          ),
          object_classes_independent=(
              'traffic_sign',
              'traffic_light',
              'street_light',
              # 'parking_meter',
              # 'mailbox',
          ),
          object_frequencies=(
              ('fence', 0.006257),
              ('pole', 0.001172),
              ('tree', 0.001924),
              ('traffic_sign', 0.000960),
              ('traffic_light', 0.000559),
              ('street_light', 0.000738),
              ('void', 0.988391),
          ),
      )
  ).lock()


def bev_localizer() -> config_dict.ConfigDict:
  return config_dict.ConfigDict(
      dict(
          bev_mapper=bev_mapper(),
          bev_mapper_query=config_dict.placeholder(config_dict.ConfigDict),
          add_confidence_query=False,
          add_confidence_map=False,
          mask_score_out_of_bounds=False,
          clip_negative_scores=True,
          add_temperature=True,
          init_temperature=2.0,
          num_pose_samples=config_dict.placeholder(int),
          num_pose_sampling_retries=1,
          query_frustum_depth=16.0,
          filter_points_in_fov=False,
          threshold_remove_accurate_poses=config_dict.placeholder(tuple),
          do_grid_refinement=False,
      )
  ).lock()


def get_config() -> config_dict.ConfigDict:
  """Dummy get_config for tests."""
  return base()
