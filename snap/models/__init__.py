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

"""Top-level register of modules and models."""
import importlib
from typing import Any, Type

import flax.linen as nn

from snap.models import base

BASEPATH = 'snap.models.{}'

MODELS = {
    'occupancy_net': ('occupancy_net', 'OccupancyNetModel'),
    'semantic_net': ('semantic_net', 'SemanticNetModel'),
    'bev_matcher': ('bev_matcher', 'BEVMatcherModel'),
    'bev_aligner': ('bev_aligner', 'BEVAlignerModel'),
    'bev_localizer': ('bev_localizer', 'BEVLocalizerModel'),
}


def get_class(modulename: str, classname: str) -> Any:
  """Import a given class in a given module."""
  modulepath = BASEPATH.format(modulename)
  return getattr(importlib.import_module(modulepath), classname)


def get_model(name: str) -> Type[base.BaseModel]:
  """Get a top-level model by name."""
  return get_class(*MODELS[name])
