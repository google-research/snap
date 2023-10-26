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

"""Common model building blocks."""
import functools
from typing import Sequence

from etils.array_types import BoolArray
from etils.array_types import FloatArray
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections

import snap.configs.defaults as default_configs

Axis = None | int | Sequence[int]


def masked_mean(x: FloatArray, mask: BoolArray, axis: Axis) -> FloatArray:
  """Like jnp.mean(x, where=mask) but returns zero when the mask is empty."""
  div = jnp.sum(jnp.where(mask.any(axis, keepdims=True), mask, True), axis)
  return jnp.sum(x * mask, axis) / div


def masked_softmax(x: FloatArray, mask: BoolArray, axis: Axis) -> FloatArray:
  """Softmax with masked values and always-finite outputs and gradients."""
  valid = mask.any(axis=axis, keepdims=True)
  mask = jnp.where(valid, mask, True)
  x = jnp.where(mask, x, -jnp.inf)
  return jax.nn.softmax(x, axis=axis)


def normalize(x: FloatArray, axis: Axis = -1, eps: float = 1e-5) -> FloatArray:
  """Normalize a vector by its L2 norm."""
  x_ = x.astype(jnp.float32)
  invalid = jnp.linalg.norm(x_, axis=axis, keepdims=True) < eps
  # We use the double-where trick to avoid NaN gradients if the norm is zero.
  y = jnp.where(invalid, eps, x_)
  z = x_ / jnp.linalg.norm(y, axis=axis, keepdims=True)
  return jnp.where(invalid, 0, z.astype(x.dtype))


class MLP(nn.Module):
  """A simple MLP."""

  config: ml_collections.ConfigDict
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(
      self,
      input_feats: FloatArray['... N'],
      train: bool = False,
  ) -> FloatArray['... D']:
    dense_layer = functools.partial(
        nn.Dense,
        kernel_init=jax.nn.initializers.glorot_uniform(),
        param_dtype=self.dtype,
    )
    activation = getattr(nn, self.config.activation)
    x = input_feats
    for i, d in enumerate(self.config.layers):
      if i > 0 or self.config.apply_input_activation:
        x = activation(x)
      x = dense_layer(d)(x)
    return x

  @classmethod
  @property
  def default_config(cls) -> ml_collections.ConfigDict:
    return default_configs.mlp()
