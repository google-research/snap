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

"""Common types and containers for model outputs."""
from typing import List, Optional

import chex
from etils.array_types import BoolArray
from etils.array_types import FloatArray


@chex.dataclass
class FeatureVolume:
  """A 3D volume of features with validity mask."""

  features: FloatArray['X Y Z D']
  valid: Optional[BoolArray['X Y Z']] = None


@chex.dataclass
class FeaturePlane:
  """A 2D plane of features with validity mask."""

  features: FloatArray['X Y D']
  valid: Optional[BoolArray['X Y']] = None


@chex.dataclass
class FeatureImagePyramid:
  """A pyramid of 2D image features with associated stride w.r.t. the input."""

  features: List[FloatArray['... H W N']]
  strides: List[FloatArray['2']]


@chex.dataclass
class LidarRaySamples:
  """Point sampled along lidar rays."""

  points: FloatArray['... 3']
  labels: BoolArray['...']
  valid: BoolArray['...']


@chex.dataclass
class OccupancySamples:
  """Occupancy values at given sample 3D points."""

  values: FloatArray['...']
  valid: BoolArray['...']
  logits: FloatArray['...']
