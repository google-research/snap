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

"""Occupancy Network config."""
from ml_collections import config_dict

from snap.configs import defaults


def get_config() -> config_dict.ConfigDict:
  """Return the config."""
  config = defaults.base()
  config.model_name = 'occupancy_net'
  config.model = defaults.occupancy_net()
  config.model.occupancy_mlp.layers = (128, 256, 1)

  config.model.scene_encoder.pretrained_xid = None
  with config.optimizer_configs.unlocked():
    config.optimizer_configs.freeze_params_reg_exp = r'scene_encoder/'

  cities = defaults.DATA_SPLITS_CITIES['train']
  locations = ','.join(
      f'{c}-n14_streetside_sceneviewpair_20views_trekkerquery' for c in cities
  )
  config.data = defaults.streetview_singlescene()
  config.data.update(
      dict(
          version='1.5.0',
          locations=dict(training=locations),
          voxel_size=0.2,
          add_lidar_rays=True,
          num_rays=10_000,
          evaluation_size=4_096,
          training_size_per_builder=5_000_000,
      )
  )
  config.batch_size = 1
  config.lr_configs.base_learning_rate = 5e-5
  config.num_training_steps = 50_000
  config.checkpoint_steps = 10_000
  config.log_summary_steps = 1_000
  config.log_eval_steps = 5_000
  config.dtype_str = 'float16'
  return config
