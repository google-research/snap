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

"""Utilities related to managing experiment configurations."""
import copy
from typing import Any
from ml_collections import config_dict


def config_update(self: config_dict.ConfigDict, other: config_dict.ConfigDict):
  """Patch around ConfigDict.update to support dict fields init as None."""
  iteritems_kwargs = {}
  if isinstance(other, config_dict.ConfigDict):
    iteritems_kwargs['preserve_field_references'] = True
  for key, value in other.iteritems(**iteritems_kwargs):
    if key not in self or value is None:
      self[key] = value
    elif isinstance(value_self := self._fields[key], config_dict.ConfigDict):
      config_update(self[key], value)
    elif isinstance(value_self, config_dict.FieldReference) and isinstance(
        value, config_dict.FieldReference
    ):
      if value.get() is not None:
        type_ = type(value_self)
        if isinstance(value_self, config_dict.FieldReference):
          type_ = value_self.get_type()
        if value.get_type() != type_:
          raise TypeError(
              'Cannot update a FieldReference from another FieldReference for'
              f' key {key}: mismatched types ({type_} vs {value.get_type()})'
          )
      self[key] = value
    else:
      self[key] = value


def configs_merge(
    a: config_dict.ConfigDict, b: config_dict.ConfigDict
) -> config_dict.ConfigDict:
  """Equivalent to (a | b) for ConfigDicts."""
  a = copy.deepcopy(a)
  config_update(a, b)
  return a


def config_diff(
    a: config_dict.ConfigDict, b: config_dict.ConfigDict
) -> dict[str, Any]:
  """Find the difference between two configurations."""
  keys = set(a.keys() + b.keys())
  diff = {}
  for key in keys:
    va = a.get(key)
    vb = b.get(key)
    if va == vb:
      continue
    if isinstance(va, config_dict.ConfigDict) and isinstance(
        vb, config_dict.ConfigDict
    ):
      d = config_diff(va, vb)
      if d:
        diff[key] = d
    else:
      diff[key] = (va, vb)
  return diff
