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

"""BEV Alignment config."""
from ml_collections import config_dict

from snap.configs import defaults


def get_config() -> config_dict.ConfigDict:
  """Return the config."""
  config = defaults.base()
  config.model_name = 'bev_aligner'
  config.model = defaults.bev_aligner()

  config.model.bev_estimator.scene_encoder.do_weighted_fusion = True
  config.model.bev_estimator.scene_encoder.top_k_view_selection = 4

  config.model.bev_estimator.pooling.pooling = 'max'
  config.model.num_pose_samples = 5000
  config.model.pose_selection = 'ransac'
  config.model.mask_score_out_of_bounds = False
  config.model.clip_negative_scores = True
  config.model.matching_dim = 32
  config.model.normalize_features = True
  config.model.add_temperature = True

  config.data = defaults.streetview_singlescene()
  config.data.update(
      dict(
          locations=dict(training='paris_streetside_scenepairs'),
          voxel_size=0.2,
          add_lidar_rays=False,
      )
  )
  config.batch_size = 1
  config.lr_configs.base_learning_rate = 1e-4
  config.num_training_epochs = 5
  config.checkpoint_steps = 10_000
  config.log_summary_steps = 1_000
  config.log_eval_steps = 5_000
  return config
