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

"""Generic wrapper for extracting feature maps from images."""
import functools
from typing import Any, Callable, List, Optional

from etils.array_types import FloatArray
import flax.linen as nn
import jax
from jax.nn import initializers
import jax.numpy as jnp
import ml_collections
import numpy as np

import snap.configs.defaults as default_configs
from snap.models import resnet
from snap.models import types


def pad_to_multiple(
    images: FloatArray['B H W N'], stride: int
) -> FloatArray['B H2 W2 N']:
  """Pad a batch of images such as their size is a multiple of stride."""
  shape = images.shape[-3:-1]
  pad = stride - np.array(shape) % stride
  padded = jnp.pad(images, [(0, 0), (0, pad[0]), (0, pad[1]), (0, 0)])
  return padded


class FPNDecoder(nn.Module):
  """Feature Pyramid Network-like decoder."""

  output_dim: int
  num_levels: int
  activation: str = 'relu'
  norm: Optional[str] = 'bit_resnet'
  kernel_init: Callable[..., Any] = initializers.lecun_normal()
  bias_init: Callable[..., Any] = initializers.zeros
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(
      self, input_features: List[FloatArray['... H W D']], train: bool = False
  ) -> List[FloatArray['... H W N']]:
    assert len(input_features) == self.num_levels
    activation = getattr(nn, self.activation)
    norm = None
    if self.norm == 'bit_resnet':
      norm = functools.partial(resnet.GroupNorm, dtype=self.dtype)
      # We stick to the standard conv because it is here an output.
    elif self.norm == 'batch_norm':
      norm = functools.partial(
          nn.BatchNorm, axis_name='batch', axis=-1, param_dtype=self.dtype
      )
    elif self.norm is not None:
      raise ValueError(self.norm)
    skip_conv = functools.partial(
        nn.Conv,
        features=self.output_dim,
        kernel_size=(1, 1),
        use_bias=norm is None,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        param_dtype=self.dtype,
    )

    out_features = []
    f_prev = None
    for level, f_skip in enumerate(input_features):
      f = activation(f_skip)
      if norm is not None:
        f = norm(name=f'{level}_skip_norm')(f)
      f = skip_conv(name=f'{level}_skip_conv')(f)
      if f_prev is not None:
        new_shape = f_prev.shape[:-3] + f.shape[-3:-1] + f_prev.shape[-1:]
        assert f.shape[-3] == f_prev.shape[-3] * 2, "Image heights don't match."
        assert f.shape[-2] == f_prev.shape[-2] * 2, "Image widths don't match."
        f_prev = jax.image.resize(f_prev, new_shape, 'bilinear')
        f = f + f_prev
      f_prev = f
      out_features.append(f)
    return out_features


class ImageEncoder(nn.Module):
  """Generic wrapper for extracting feature maps from images."""

  config: ml_collections.ConfigDict
  dtype: jnp.dtype = jnp.float32

  def setup(self):
    num_pyr_levels = self.config.num_pyr_levels
    if self.config.encoder_name == 'resnet':
      self.encoder = resnet.ResNetV2(self.config.encoder, self.dtype)
      if num_pyr_levels is None:
        num_pyr_levels = len(self.encoder.level_names)
      self.max_stride = (
          (not self.config.encoder.skip_root_block) * 2 + num_pyr_levels - 1
      )
    else:
      raise ValueError(self.config.encoder_name)
    self.level_names = self.encoder.level_names[:num_pyr_levels][::-1]
    self.decoder = FPNDecoder(
        self.config.output_dim, num_pyr_levels, dtype=self.dtype
    )

  def __call__(
      self,
      image: FloatArray['B H W 3'],
      train: bool = False,
  ) -> types.FeatureImagePyramid:
    image = image.astype(self.dtype)
    input_shape = np.array(image.shape[-3:-1])
    image_padded = pad_to_multiple(image, 2**self.max_stride)
    padded_shape = np.array(image_padded.shape[-3:-1])
    encoder_features = self.encoder(image_padded, train=train)

    skip_features = []
    for layer_name in self.level_names:
      # Get the last unit of each stage.
      _, f = sorted(encoder_features[layer_name].items())[-1]
      skip_features.append(f)

    out_features = self.decoder(skip_features, train=train)
    strides = [padded_shape / np.array(f.shape[-3:-1]) for f in out_features]
    out_features_crop = []
    for s, f in zip(strides, out_features):
      h, w = np.round(np.ceil(input_shape / s)).astype(int)
      out_features_crop.append(f[..., :h, :w, :])
    return types.FeatureImagePyramid(
        features=out_features_crop, strides=strides
    )

  @classmethod
  @property
  def default_config(cls) -> ml_collections.ConfigDict:
    return default_configs.image_encoder()
