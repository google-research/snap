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

"""BEV Localization config with a fixed ground plane."""
from ml_collections import config_dict

from snap.configs import bev_localization


def get_config(args_str: None | str = None) -> config_dict.ConfigDict:
  """Return the config."""
  config = bev_localization.get_config(args_str)
  bev_mappers = [config.model.bev_mapper]
  if config.model.bev_mapper_query is not None:
    bev_mappers.append(config.model.bev_mapper_query)
  for mapper in bev_mappers:
    # A single height plane 2.5 meters below the cameras.
    mapper.scene_z_offset = 2.5 + config.data.voxel_size / 2
    mapper.scene_z_height = config.data.voxel_size
    if mapper.streetview_encoder is not None:
      mapper.streetview_encoder.do_weighted_fusion = False
  return config
