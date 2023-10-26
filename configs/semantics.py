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

"""Semantic Segmentation Network config."""

from ml_collections import config_dict

from snap.configs import defaults


def get_config(args_str: None | str = None) -> config_dict.ConfigDict:
  """Return the config."""
  config = defaults.base()
  config.model_name = 'semantic_net'
  config.model = defaults.semantic_net()
  config.model.apply_random_flip = True
  config.model.decoder_dim = 256
  config.model.decoder_type = 'resnet_stage'
  config.model.resnet_num_units = 2
  config.model.bev_estimator.scene_encoder.max_view_distance = 20

  args = defaults.parse_argument_string(args_str)
  image_encoder = defaults.resnet(args['image_encoder'])
  config.model.bev_estimator.scene_encoder.image_encoder.encoder = image_encoder

  config.model.bev_estimator.pretrained_xid = None
  with config.optimizer_configs.unlocked():
    config.optimizer_configs.freeze_params_reg_exp = r'bev_estimator/'

  config.data = defaults.streetview_singlescene()
  config.data.update(
      dict(
          locations=dict(
              training=(
                  'train-v10abcde_streetside_20views_semantics_noaerial_train'
              ),
              evaluation='val-v10abcde_streetside_20views_semantics_valid',
          ),
          voxel_size=0.2,
          add_images=True,
          add_rasters=True,
          evaluation_size=1_024,
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
