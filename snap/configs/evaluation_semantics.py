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
          split='val-v10abcde',
          name_pattern='{}_streetside_20views_semantics_eval',
          loader=defaults.streetview_singlescene(),
      ),
      model=config_dict.create(),
  )
  config.data.loader.evaluation_size = 10_000
  return config.lock()
