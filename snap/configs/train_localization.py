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

"""BEV Localization config."""
from ml_collections import config_dict

from snap.configs import defaults


def get_config(args_str: None | str = None) -> config_dict.ConfigDict:
  """Return the config."""
  config = defaults.base()
  config.model_name = 'bev_localizer'
  config.model = defaults.bev_localizer()
  config.model.filter_points_in_fov = True
  config.model.num_pose_samples = 10_000
  # This improves the training metrics but it's not clear yet whether this also
  # improves the evaluation results.
  config.model.num_pose_sampling_retries = 8

  args = defaults.parse_argument_string(args_str)
  image_encoder = defaults.resnet(args['image_encoder'])

  map_modalities = args['modalities'].split('+')
  config.model.bev_mapper = defaults.bev_mapper(map_modalities)

  if defaults.MapModalities.STREETVIEW in map_modalities:
    config.model.bev_mapper.streetview_encoder.image_encoder.encoder = (
        image_encoder
    )
  else:
    mapper_query = defaults.bev_mapper(
        modalities=(defaults.MapModalities.STREETVIEW,)
    )
    mapper_query.streetview_encoder.image_encoder.encoder = image_encoder
    # Make the fusion MLP a bit deeper
    dim = mapper_query.streetview_encoder.feature_dim
    mapper_query.streetview_encoder.fusion.layers = (dim * 2, dim * 2, dim)
    config.model.bev_mapper_query = mapper_query

  cities = defaults.DATA_SPLITS_CITIES['train']
  locations = ','.join(
      f'{c}-n14_streetside_sceneviewpair_20views_trekkerquery' for c in cities
  )
  config.data = defaults.streetview_singlescene()
  config.data.update(
      dict(
          # Make sure that we load the correct dataset version. We should update
          # this once the version in data/builder.py is updated and generated.
          version='1.5.0',
          locations=dict(training=locations),
          voxel_size=0.2,
          add_images=defaults.MapModalities.STREETVIEW in map_modalities,
          add_rasters=(
              (defaults.MapModalities.AERIAL in map_modalities)
              or (defaults.MapModalities.SEMANTIC in map_modalities)
          ),
          evaluation_size=8_192,
          training_size_per_builder=5_000_000,
      )
  )
  config.batch_size = 1

  # The large model is much slower so we checkpoint and evaluate more often.
  if args['image_encoder'] == 'R152x2':
    config.checkpoint_steps = 2_000
    config.log_summary_steps = 500
    config.log_eval_steps = 4_000
    config.num_training_steps = 200_000
  else:
    config.checkpoint_steps = 10_000
    config.log_summary_steps = 1_000
    config.log_eval_steps = 5_000
    config.num_training_steps = 400_000

  config.lr_configs.start_decay_step = config.get_ref('num_training_steps') // 2
  config.lr_configs.base_learning_rate = 5e-5
  config.lr_configs.factors = 'constant * cosine_decay'
  config.lr_configs.steps_per_cycle = config.get_ref(
      'num_training_steps'
  ) - config.lr_configs.get_ref('start_decay_step')
  config.dtype_str = 'float16'
  return config
