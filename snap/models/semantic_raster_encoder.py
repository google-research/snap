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

"""Encode a 2D semantic map, as multichannel rasters, into a neural map."""
from etils.array_types import BoolArray
import flax.linen as nn
import jax.numpy as jnp
import ml_collections

import snap.configs.defaults as default_configs
from snap.data import types as data_types
from snap.models import image_encoder
from snap.models import types


class SemanticRasterEncoder(nn.Module):
  """Encode 2D semantic rasters into a neural map."""

  config: ml_collections.ConfigDict
  raster_classes: tuple[str, ...]
  dtype: jnp.dtype = jnp.float32

  def __post_init__(self):
    # Surfel road classes are mutually exclusive so we treat them as a
    # multiclass label. Other classes are independent so they are instead
    # binary labels.
    self.indices_surfel_road = []
    self.indices_other_classes = []
    for i, c in enumerate(self.raster_classes):
      (
          self.indices_surfel_road
          if c in data_types.SURFEL_ROAD_CLASSES
          else self.indices_other_classes
      ).append(i)
    super().__post_init__()

  def setup(self):
    self.encoder = image_encoder.ImageEncoder(self.config.encoder)
    self.embeddings_surfel_road = nn.Embed(
        len(self.indices_surfel_road),
        self.config.embedding_dim,
        param_dtype=self.dtype,
    )
    self.embeddings_other_classes = nn.Embed(
        len(self.indices_other_classes) * 2,  # each class has labels 0 or 1
        self.config.embedding_dim,
        param_dtype=self.dtype,
    )

  def __call__(
      self, rasters: BoolArray['B H W N'], train: bool = False
  ) -> types.FeatureImagePyramid:
    assert rasters.shape[-1] == len(self.raster_classes)
    rasters_surfel_roads = rasters[..., self.indices_surfel_road]
    label_surfel_roads = jnp.argmax(rasters_surfel_roads, axis=-1)
    f_surfel_roads = self.embeddings_surfel_road(label_surfel_roads)

    rasters_others = rasters[..., self.indices_other_classes]
    labels_others = jnp.arange(
        rasters_others.shape[-1]
    ) + rasters_others.astype(int)
    f_others = self.embeddings_other_classes(labels_others)
    # Flatten the last two dimensions.
    f_others = f_others.reshape(*f_others.shape[:-2], -1)

    f_rasters = jnp.concatenate([f_surfel_roads, f_others], axis=-1)
    f_pyramid = self.encoder(f_rasters)
    return f_pyramid

  @classmethod
  @property
  def default_config(cls) -> ml_collections.ConfigDict:
    return default_configs.semantic_raster_encoder()
