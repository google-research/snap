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

"""A better base model."""
from typing import Any, Callable, Dict, Optional, Tuple

from etils.array_types import FloatArray
import flax.linen as nn
import jax.numpy as jnp
import ml_collections

Batch = Dict[str, Any]
Predictions = Dict[str, Any]
LossDict = MetricsDict = Dict[str, FloatArray['batch']]
LossMetricsTuple = Tuple[LossDict, MetricsDict]
LossMetricsFn = Callable[
    [Predictions, Batch, Optional[jnp.ndarray]], LossMetricsTuple
]


class BaseModel:
  """Defines commonalities between all models."""

  def __init__(
      self,
      config: ml_collections.ConfigDict,
      dataset_meta_data: Dict[str, Any],
      dtype: jnp.dtype = jnp.float32,
  ):
    self.config = config
    self.dataset_meta_data = dataset_meta_data
    self.dtype = dtype
    self.flax_model = self.build_flax_model()

  def loss_metrics_function(
      self,
      pred: Predictions,
      batch: Batch,
      model_params: Optional[jnp.ndarray] = None,
  ) -> LossMetricsTuple:
    """Returns the loss and metric dictionaries."""
    raise NotImplementedError('Subclasses must implement metrics.')

  def build_flax_model(self) -> nn.Module:
    raise NotImplementedError('Subclasses must implement build_flax_model().')

  def default_flax_model_config(self) -> ml_collections.ConfigDict:
    """Default config for the flax model that is built in `build_flax_model`.

    This function in particular serves the testing functions and supposed to
    provide config that are passed to the flax_model when it's built in
    `build_flax_model` function, e.g., `model_dtype_str`.
    """
    raise NotImplementedError(
        'Subclasses must implement default_flax_model_config().'
    )
