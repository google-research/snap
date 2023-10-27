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

"""BEV Matching config."""
from ml_collections import config_dict

from snap.configs import defaults


def get_config() -> config_dict.ConfigDict:
  """Return the config."""
  config = defaults.base()
  config.model_name = 'bev_matcher'
  config.model = defaults.bev_matcher()

  # add depth MLP
  dim = config.model.bev_estimator.scene_encoder.image_encoder.output_dim
  depth_mlp = defaults.mlp()
  depth_mlp.apply_input_activation = True
  depth_mlp.layers = (dim // 2, dim)
  config.model.bev_estimator.scene_encoder.depth_mlp = depth_mlp

  config.data = defaults.streetview_singlescene()
  config.data.update(
      dict(
          locations=dict(training='zh-full_streetside_scenepairs'),
          voxel_size=0.2,
          add_lidar_rays=False,
      )
  )
  config.batch_size = 1
  config.lr_configs.base_learning_rate = 1e-4
  config.num_training_epochs = 1
  config.checkpoint_steps = 10_000
  config.dtype_str = 'float16'
  return config
