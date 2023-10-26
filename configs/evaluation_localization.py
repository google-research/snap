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

"""Config for evaluation."""
from ml_collections import config_dict

from snap.configs import defaults


def get_config() -> config_dict.ConfigDict:
  """Return the config."""
  config = config_dict.create(
      xid=config_dict.placeholder(int),
      wid=1,
      checkpoint_step=config_dict.placeholder(int),
      batch_size=4,
      rng_seed=0,
      dtype_str='float32',
      tag='',
      overwrite=False,
      data=config_dict.create(
          rng_seed=0,
          split='test3',
          name_pattern=(
              '{}-n14_streetside_sceneviewpair_20views_trekkerquery_eval'
          ),
          loader=defaults.streetview_singlescene(),
      ),
      model=config_dict.create(
          num_pose_samples=20_000,
          num_pose_sampling_retries=8,
          do_grid_refinement=True,
      ),
  )
  config.data.loader.evaluation_size = 4096  # We could also use all examples.
  return config.lock()
