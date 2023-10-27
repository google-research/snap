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

"""Predict a semantic raster from a BEV feature plane."""
from typing import Optional, Sequence

from etils.array_types import Array
from etils.array_types import BoolArray
from etils.array_types import FloatArray
from etils.array_types import IntArray
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax

import snap.configs.defaults as default_configs
from snap.models import base
from snap.models import bev_estimator
from snap.models import layers
from snap.models import resnet
from snap.models import types
from snap.utils import grids


def balancing_weights(
    frequencies: dict[str, float],
    classes: Sequence[str],
    binary: bool = False,
    eps: float = 1e-3,
) -> FloatArray['N'] | tuple[FloatArray['N'], FloatArray['N']]:
  """Compute the per-class weights for imbalanced classification."""
  frequencies = np.array([frequencies[c] for c in classes])
  if not binary:
    frequencies /= frequencies.sum()  # renormalize
  frequencies.clip(min=eps, out=frequencies)
  weights = jnp.asarray(1 / (frequencies * len(classes)))
  if binary:  # negative weights
    weights_neg = 1 / ((1 - frequencies).clip(min=eps) * len(classes))
    return weights, jnp.asarray(weights_neg)
  return weights


def multiclass_crossentropy_metrics(
    logits: FloatArray['B H W N'],
    labels: IntArray['B H W'],
    valid: BoolArray['B H W'],
    classes: Sequence[str],
    frequencies: None | dict[str, float],
    namespace: None | str = None,
) -> tuple[FloatArray['B'], dict[str, FloatArray['B']]]:
  """Compute multiclass cross-entropy loss and metrics."""
  nll = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
  if frequencies:
    weights = balancing_weights(dict(frequencies), classes)
    nll = nll * weights[..., labels]  # B,H,W
    assert nll.ndim == 3
  nll = layers.masked_mean(nll, valid, (1, 2))

  mask = labels[..., None] == jnp.arange(logits.shape[-1])
  correct = jnp.argmax(logits, axis=-1) == labels
  acc = layers.masked_mean(correct, valid, (1, 2))
  recall = layers.masked_mean(
      correct[..., None], valid[..., None] & mask, (1, 2)
  )
  suffix = f'/{namespace}' if namespace else ''
  metrics = {
      f'accuracy{suffix}': acc,
      f'recall/average{suffix}': recall.mean(-1),
  }
  for i, c in enumerate(classes):
    metrics[f'recall/{c}'] = recall[..., i]
  return nll, metrics


def binary_crossentropy_metrics(
    logits: FloatArray['B H W N'],
    gt_mask: BoolArray['B H W N'],
    valid: BoolArray['B H W'],
    classes: Sequence[str],
    frequencies: None | dict[str, float],
    namespace: None | str = None,
) -> tuple[FloatArray['B'], dict[str, FloatArray['B']]]:
  """Compute binary cross-entropy loss and metrics."""
  nll = optax.sigmoid_binary_cross_entropy(logits, gt_mask)
  if frequencies:
    w_pos, w_neg = balancing_weights(dict(frequencies), classes, binary=True)
    nll = nll * jnp.where(gt_mask, w_pos, w_neg)
    assert nll.ndim == 4
  nll = layers.masked_mean(nll.mean(-1), valid, (1, 2))

  correct = (jax.nn.sigmoid(logits) > 0.5) == gt_mask
  recall = layers.masked_mean(correct, valid[..., None] & gt_mask, (1, 2))
  suffix = f'/{namespace}' if namespace else ''
  metrics = {f'recall/average{suffix}': recall.mean(-1)}
  for i, c in enumerate(classes):
    metrics[f'recall/{c}'] = recall[..., i]
  return nll, metrics


@jax.vmap
def batched_raster_flip(
    raster: Array['H W ...'], flip_mask: BoolArray['2']
) -> Array['H W ...']:
  """Flip the two spatial dimensions of a raster."""
  for i in range(2):
    raster = jnp.where(flip_mask[i], jnp.flip(raster, axis=i), raster)
  return raster


class SemanticNet(nn.Module):
  """Predict per-point occupancy from an image-based scene representation."""

  config: ml_collections.ConfigDict
  grid: grids.Grid2D
  dtype: jnp.dtype = jnp.float32

  def setup(self):
    self.bev_estimator = bev_estimator.BEVEstimator(
        config=self.config.bev_estimator,
        grid=self.grid,
        dtype=self.dtype,
    )

    num_classes = len(self.config.area_classes)
    self.object_classes = (
        self.config.object_classes_exclusive
        + self.config.object_classes_independent
    )
    if self.object_classes:
      num_classes += len(self.object_classes) + 1  # void

    dim = self.config.decoder_dim
    match self.config.decoder_type:
      case 'mlp':
        mlp_config = default_configs.mlp()
        mlp_config.layers = (dim,) * self.config.mlp_num_layers + (num_classes,)
        self.decoder = layers.MLP(mlp_config, self.dtype)
      case 'resnet_stage':
        mlp_config = default_configs.mlp()
        mlp_config.layers = (dim, num_classes)
        self.decoder = nn.Sequential([
            nn.Dense(
                dim,
                param_dtype=self.dtype,
                kernel_init=jax.nn.initializers.glorot_uniform(),
            ),
            resnet.ResNetStage(self.config.resnet_num_units, dtype=self.dtype),
            lambda out, _: out,  # Select the final output.
            layers.MLP(mlp_config, self.dtype),
        ])
      case _:
        raise ValueError(f'Unknown {self.config.decoder_type}')

  def __call__(
      self, data: base.Batch, train: bool = False, debug: bool = False
  ) -> base.Predictions:
    if 'map' in data:
      data = data['map']
    pred = self.bev_estimator(data, train)
    neural_map = pred['bev_features']

    flips = None
    if train and self.config.apply_random_flip:
      flips = jax.random.bernoulli(
          self.make_rng('sampling'),
          shape=(len(neural_map.features), 2),  # batch x number of axes
      )
      neural_map = types.FeaturePlane(
          features=batched_raster_flip(neural_map.features, flips),
          valid=batched_raster_flip(neural_map.valid, flips),
      )
    logits = self.decoder(neural_map.features).astype(jnp.float32)
    logits = jnp.where(neural_map.valid[..., None], logits, 0)
    if flips is not None:
      logits = batched_raster_flip(logits, flips)
    pred['logits_areas'], logits = jnp.split(
        logits, [len(self.config.area_classes)], axis=-1
    )
    if self.object_classes:
      excl, indep = jnp.split(
          logits, [len(self.config.object_classes_exclusive) + 1], axis=-1
      )
      pred['logits_objects_exclusive'] = excl
      pred['logits_objects_independent'] = indep
    return pred

  @classmethod
  @property
  def default_config(cls) -> ml_collections.ConfigDict:
    return default_configs.semantic_net()


class SemanticNetModel(base.BaseModel):
  """Trainer-facing wrapper for the SemanticNet."""

  def build_flax_model(self) -> nn.Module:
    return SemanticNet(
        self.config,
        self.dataset_meta_data['grid'].bev(),
        self.dtype,
    )

  @classmethod
  def default_flax_model_config(cls) -> ml_collections.ConfigDict:
    return SemanticNet.default_config  # pytype: disable=bad-return-type

  @property
  def gt_indices(self) -> dict[str, int]:
    gt_classes = self.dataset_meta_data['semantic_classes_gt']
    return {c: i for i, c in enumerate(gt_classes)}

  def transfer_labels_from_pcm(
      self, masks: BoolArray['... N'], masks_pcm: BoolArray
  ) -> BoolArray['... N']:
    indices_gt = self.gt_indices
    indices_pcm = {
        c: i
        for i, c in enumerate(self.dataset_meta_data['semantic_map_classes'])
    }
    class_names = [
        ('building', 'buildings_raw'),
        ('tree', 'tree'),
    ]
    for name_gt, name_pcm in class_names:
      if name_gt in indices_gt and name_pcm in indices_pcm:
        mask_pcm = masks_pcm[..., indices_pcm[name_pcm]]
        masks = masks.at[..., indices_gt[name_gt]].set(mask_pcm)
    return masks

  def _create_exclusive_labels(
      self,
      masks_all: BoolArray['... N'],
      classes: Sequence[str],
      add_void: bool = False,
  ) -> tuple[IntArray['...'], BoolArray['...']]:
    gt_indices = self.gt_indices
    indices = [gt_indices[c] for c in classes]
    masks = masks_all[..., jnp.array(indices)]

    # Group all line labels into a single class.
    if 'line' in classes:
      mask_line = masks_all[..., gt_indices['line']]
      for c in ['stopline', 'otherlanemarking']:
        if c in gt_indices and c not in classes:
          mask_line = mask_line | masks_all[..., gt_indices[c]]
      masks = masks.at[..., list(classes).index('line')].set(mask_line)
    valid = masks.any(axis=-1)
    labels = jnp.argmax(masks, axis=-1)
    if add_void:
      labels = jnp.where(valid, labels, len(classes))
    return labels, valid

  def create_area_labels(
      self, masks_all: BoolArray['... N']
  ) -> tuple[IntArray['...'], BoolArray['...']]:
    return self._create_exclusive_labels(masks_all, self.config.area_classes)

  def create_object_labels(
      self, masks: BoolArray['... N']
  ) -> tuple[IntArray['...'], BoolArray['... M']]:
    labels_excl, _ = self._create_exclusive_labels(
        masks, self.config.object_classes_exclusive, add_void=True
    )

    gt_indices = self.gt_indices
    indices_indep = [
        gt_indices[c] for c in self.config.object_classes_independent
    ]
    masks_indep = masks[..., jnp.array(indices_indep)]

    return labels_excl, masks_indep

  def _loss_metrics_areas(self, pred, masks):
    labels, valid = self.create_area_labels(masks)
    valid = pred['bev_features'].valid & valid
    nll, metrics = multiclass_crossentropy_metrics(
        pred['logits_areas'],
        labels,
        valid,
        self.config.area_classes,
        dict(self.config.area_frequencies or []),
    )
    return nll, metrics

  def _loss_metrics_objects(self, pred, masks):
    labels_excl, masks_indep = self.create_object_labels(masks)
    nll_excl, metrics_excl = multiclass_crossentropy_metrics(
        pred['logits_objects_exclusive'],
        labels_excl,
        pred['bev_features'].valid,
        (*self.config.object_classes_exclusive, 'void'),
        dict(self.config.object_frequencies or []),
        namespace='excl',
    )
    nll_indep, metrics_indep = binary_crossentropy_metrics(
        pred['logits_objects_independent'],
        masks_indep,
        pred['bev_features'].valid,
        self.config.object_classes_independent,
        dict(self.config.object_frequencies or []),
        namespace='indep',
    )
    return nll_excl, nll_indep, metrics_excl | metrics_indep

  def loss_metrics_function(
      self,
      pred: base.Predictions,
      data: base.Batch,
      model_params: Optional[jnp.ndarray] = None,
  ) -> base.LossMetricsTuple:
    if 'map' in data:
      data = data['map']
    masks = data['rasters']['gt_semantics']
    masks = self.transfer_labels_from_pcm(masks, data['rasters']['semantics'])
    nll_areas, metrics = self._loss_metrics_areas(pred, masks)
    losses = {'nll_areas': nll_areas}
    total = nll_areas

    if 'logits_objects_exclusive' in pred:
      nll_excl, nll_indep, metrics_objects = self._loss_metrics_objects(
          pred, masks
      )
      total = (total + (nll_excl + nll_indep) / 2) / 2
      losses['nll_objects_exclusive'] = nll_excl
      losses['nll_objects_indep'] = nll_indep
      metrics |= metrics_objects

    losses['total'] = total
    metrics = {f'semantics/{k}': v for k, v in metrics.items()}
    return losses, metrics

  def pack_evaluation_metrics(
      self,
      training_metrics: base.MetricsDict,
      losses: base.LossDict,
      data: base.Batch,
      pred: base.Predictions,
  ) -> base.MetricsDict:
    if 'map' in data:
      data = data['map']
    gt_classes = self.dataset_meta_data['semantic_classes_gt']
    gt_counts = data['rasters']['gt_semantics'].sum(axis=(-3, -2))
    gt_counts = {
        f'gt_counts/{c}': gt_counts[..., i] for i, c in enumerate(gt_classes)
    }
    metrics = training_metrics | dict(loss=losses['total']) | gt_counts
    return metrics
