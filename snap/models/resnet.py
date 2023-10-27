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

"""ResNet implementation from Big Transfer:
       Big Transfer (BiT): General Visual Representation Learning
       A. Kolesnikov, L. Beyer, X. Zhai, J. Puigcerver, J. Yung, S. Gelly, N. Houlsby
       ECCV 2020
   Copied to support mixed-precision training.
   https://github.com/google-research/big_vision/blob/main/big_vision/models/bit_paper.py
"""
import functools
from typing import Optional, Sequence, Any, Dict

from absl import logging
from big_vision.models import bit
from big_vision.models import bit_paper
from etils.array_types import FloatArray
import flax.linen as nn
import jax.numpy as jnp
import ml_collections

import snap.configs.defaults as default_configs


def standardize(
    x: FloatArray, axis: None | int | Sequence[int], eps: float
) -> FloatArray:
  dtype = x.dtype
  x = x.astype(jnp.float32)
  x = x - jnp.mean(x, axis=axis, keepdims=True)
  x = x / jnp.sqrt(jnp.mean(jnp.square(x), axis=axis, keepdims=True) + eps)
  return x.astype(dtype)


# Defined our own, because we compute normalizing variance slightly differently,
# which does affect performance when loading pre-trained weights!
class GroupNorm(nn.Module):
  """Group normalization (arxiv.org/abs/1803.08494)."""

  ngroups: int = 32
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x: FloatArray['... N']) -> FloatArray['... N']:
    input_shape = x.shape
    group_shape = x.shape[:-1] + (self.ngroups, x.shape[-1] // self.ngroups)

    x = x.reshape(group_shape)

    # Standardize along spatial and group dimensions
    x = standardize(x, axis=[1, 2, 4], eps=1e-5)
    x = x.reshape(input_shape)

    bias_scale_shape = tuple([1, 1, 1] + [input_shape[-1]])
    x = x * self.param(
        'scale', nn.initializers.ones, bias_scale_shape, self.dtype
    )
    x = x + self.param(
        'bias', nn.initializers.zeros, bias_scale_shape, self.dtype
    )
    return x


class StdConv(nn.Conv):

  def param(self, name, *a, **kw):
    param = super().param(name, *a, **kw)
    if name == 'kernel':
      param = standardize(param, axis=[0, 1, 2], eps=1e-10)
    return param


class RootBlock(nn.Module):
  """Root block of ResNet."""

  width: int
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x):
    x = StdConv(
        self.width,
        (7, 7),
        (2, 2),
        padding=[(3, 3), (3, 3)],
        use_bias=False,
        name='conv_root',
        param_dtype=self.dtype,
    )(x)
    x = nn.max_pool(x, (3, 3), strides=(2, 2), padding=[(1, 1), (1, 1)])
    return x


class ResidualUnit(nn.Module):
  """Bottleneck ResNet block."""

  nmid: Optional[int] = None
  strides: Sequence[int] = (1, 1)
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x):
    nmid = self.nmid or x.shape[-1] // 4
    nout = nmid * 4
    conv = functools.partial(StdConv, use_bias=False, param_dtype=self.dtype)
    norm = functools.partial(GroupNorm, dtype=self.dtype)

    residual = x
    x = norm(name='gn1')(x)
    x = nn.relu(x)

    if x.shape[-1] != nout or self.strides != (1, 1):
      residual = conv(nout, (1, 1), self.strides, name='conv_proj')(x)

    x = conv(nmid, (1, 1), name='conv1')(x)
    x = norm(name='gn2')(x)
    x = nn.relu(x)
    x = conv(
        nmid, (3, 3), self.strides, padding=[(1, 1), (1, 1)], name='conv2'
    )(x)
    x = norm(name='gn3')(x)
    x = nn.relu(x)
    x = conv(nout, (1, 1), name='conv3')(x)

    return x + residual


class ResNetStage(nn.Module):
  """A stage (sequence of same-resolution blocks)."""

  block_size: int
  nmid: Optional[int] = None
  first_stride: Sequence[int] = (1, 1)
  checkpoint_units: bool = False
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x):
    out = {}
    unit = functools.partial(ResidualUnit, nmid=self.nmid, dtype=self.dtype)
    if self.checkpoint_units:
      unit = nn.remat(unit)
    x = out['unit01'] = unit(strides=self.first_stride, name='unit01')(x)
    for i in range(1, self.block_size):
      x = out[f'unit{i+1:02d}'] = unit(name=f'unit{i+1:02d}')(x)
    return x, out


class ResNetV2(nn.Module):
  """BiT variant."""

  config: ml_collections.ConfigDict
  dtype: jnp.dtype = jnp.float32

  def __post_init__(self):
    self.blocks = bit.get_block_desc(self.config.depth)
    if self.config.limit_num_blocks is not None:
      self.blocks = self.blocks[: self.config.limit_num_blocks]
    self.level_names = [f'stage{i + 1}' for i in range(len(self.blocks))]
    super().__post_init__()

  @nn.compact
  def __call__(self, image, *, train=False):
    width = int(64 * self.config.width)

    root_block = functools.partial(RootBlock, dtype=self.dtype)
    stage = functools.partial(
        ResNetStage,
        dtype=self.dtype,
        checkpoint_units=self.config.checkpoint_units,
    )
    if self.config.checkpoint_blocks:
      root_block = nn.remat(root_block)
      if not self.config.checkpoint_units:
        stage = nn.remat(stage)

    out = {}
    x = image * 2 - 1  # big_vision normalizes to [-1,1]
    if self.config.skip_root_block:
      x = StdConv(
          width,
          (3, 3),
          padding=[(1, 1)] * 2,
          use_bias=False,
          name='conv_root',
          param_dtype=self.dtype,
      )(x)
    else:
      x = out['stem'] = root_block(width=width, name='root_block')(x)
    x, out['stage1'] = stage(self.blocks[0], nmid=width, name='block1')(x)
    for i, block_size in enumerate(self.blocks[1:], 1):
      x, out[f'stage{i + 1}'] = stage(
          block_size, width * 2**i, first_stride=(2, 2), name=f'block{i + 1}'
      )(x)
    return out

  @classmethod
  @property
  def default_config(cls) -> ml_collections.ConfigDict:
    return default_configs.resnet()

  def load_pretrained_variables(self) -> None | Dict[str, Any]:
    if self.config.pretrained_path is None:
      return
    logging.info(
        'Loading pretrained weights for %s from %s.',
        self.__class__.__name__,
        self.config.pretrained_path,
    )
    params = bit_paper.u.load_params(None, self.config.pretrained_path)
    return {'params': params}
