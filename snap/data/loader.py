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

"""Training and evaluation loader for the StreetView dataset."""
import collections
import functools
import itertools
import os
from typing import Callable, Iterator, Optional

from absl import logging
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.dataset_lib import dataset_utils
from scenic.dataset_lib import datasets
import tensorflow as tf
import tensorflow_datasets as tfds

import snap.configs.defaults as default_configs
from snap.data import types
from snap.data.types import DataDict
from snap.utils import geometry
from snap.utils import grids


def prefetch_to_device_with_strings(
    iterator: Iterator[DataDict],
    size: int,
    devices: None | list[jax.Device] = None,
) -> Iterator[DataDict]:
  """Like flax.jax_utils.prefetch_to_device but exclude strings."""
  queue = collections.deque()
  devices = devices or jax.local_devices()

  def _prefetch(xs):
    if xs.dtype.type is np.str_:
      return xs
    return jax.device_put_sharded(list(xs), devices)

  def enqueue(n):  # Enqueues *up to* `n` elements from the iterator.
    for data in itertools.islice(iterator, n):
      queue.append(jax.tree_util.tree_map(_prefetch, data))

  enqueue(size)  # Fill up the buffer.
  while queue:
    yield queue.popleft()
    enqueue(1)


def pad_lidar_rays(rays: dict[str, tf.Tensor], num_target: int) -> DataDict:
  """Pad the lidar rays to a fixed number."""
  num = tf.shape(rays['points'])[0]
  num_sampled = tf.minimum(num, num_target)
  indices = tf.random.shuffle(tf.range(num))[:num_sampled]
  points = tf.gather(rays['points'], indices)
  origins = tf.gather(rays['origins'], indices)
  missing = num_target - num_sampled
  rays_padded = {
      'points': tf.pad(points, [[0, missing], [0, 0]]),
      'origins': tf.pad(origins, [[0, missing], [0, 0]]),
      'mask': tf.pad(tf.ones(num_sampled, tf.bool), [[0, missing]]),
  }
  if 'semantics' in rays:
    semantics = tf.gather(rays['semantics'], indices)
    rays_padded['semantics'] = tf.pad(semantics, [[0, missing]])
  return rays_padded


def process_scene_example(
    example: DataDict,
    config: ml_collections.ConfigDict,
    dtype: tf.dtypes.DType,
    is_single_view: bool = False,
) -> DataDict:
  """Process one TF example scene."""
  ret = {
      'T_view2scene': example['views']['T_camera2scene'],
      'camera': example['views']['intrinsics'],
      'scene_id': example['scene_id'],
      'vehicle_type': example['vehicle_type'],
      'latlng': example['coordinates']['center_latlng'],
  }
  if config.add_images or is_single_view:
    images = example['views']['color_image']
    images = dataset_utils.normalize(images, dtype)
    images = tf.cast(images, dtype)
    ret['images'] = images
  if config.add_lidar_rays and not is_single_view:
    ret['lidar_rays'] = pad_lidar_rays(
        example['point_cloud']['rays'], config.num_rays
    )
  if config.add_rasters and not is_single_view:
    ret['rasters'] = example['rasters']
    if 'rgb' in example['rasters']:
      rgb = tf.cast(dataset_utils.normalize(example['rasters']['rgb']), dtype)
      ret['rasters']['rgb'] = rgb
  return ret


def process_example(
    example: DataDict,
    config: ml_collections.ConfigDict,
    dtype: tf.dtypes.DType = tf.float32,
) -> DataDict:
  """Preprocesses one TF example."""
  # TODO: Switch to match-case once b/270274562 is resolved.
  if config.mode == types.DataMode.SINGLE_SCENE:
    item = process_scene_example(example, config, dtype)
  elif config.mode == types.DataMode.PAIR_SCENES:
    item = {k: example[k] for k in ('T_j2i', 'overlap', 'time_delta_days')}
    for k in ('scene_i', 'scene_j'):
      item[k] = process_scene_example(example[k], config, dtype)
  elif config.mode == types.DataMode.PAIR_SCENE_VIEW:
    item = {
        k: example[k]
        for k in ('T_query2map', 'overlap', 'time_delta_days', 'pair_id')
    }
    for k in ('map', 'query'):
      item[k] = process_scene_example(
          example[k], config, dtype, is_single_view=(k == 'query')
      )
  else:
    raise ValueError(config.mode)
  return item


def process_scene_batch(batch: DataDict) -> DataDict:
  cameras = geometry.FisheyeCamera.from_dict(batch['camera'])
  tfm_view2scene = geometry.Transform3D(**batch['T_view2scene'])
  batch.update({
      'T_view2scene': tfm_view2scene,
      'camera': cameras,
      'scene_id': np.asarray(batch['scene_id']).astype(str),
      'vehicle_type': np.asarray(batch['vehicle_type']).astype(str),
  })
  return batch


def process_batch(
    batch: DataDict, config: ml_collections.ConfigDict
) -> DataDict:
  """Process a jax batch."""
  # TODO: Switch to match-case.
  if config.mode == types.DataMode.SINGLE_SCENE:
    batch = process_scene_batch(batch)
  elif config.mode == types.DataMode.PAIR_SCENES:
    for k in ('scene_i', 'scene_j'):
      batch[k] = process_scene_batch(batch[k])
    batch['T_j2i'] = geometry.Transform3D(**batch['T_j2i'])
  elif config.mode == types.DataMode.PAIR_SCENE_VIEW:
    for k in ('map', 'query'):
      batch[k] = process_scene_batch(batch[k])
    batch['T_query2map'] = geometry.Transform3D(**batch['T_query2map'])
    batch['pair_id'] = np.asarray(batch['pair_id']).astype(str)
  else:
    raise ValueError(config.mode)
  return batch


def get_dummy_batch(
    batch_size: int,
    builder: tfds.core.DatasetBuilder,
    example_fn: Callable[[DataDict], DataDict],
    batch_fn: Callable[[DataDict], DataDict],
) -> DataDict:
  """Get a single batch for model init."""
  ds = builder.as_dataset(split='train')
  ds = ds.map(example_fn)
  ds = ds.batch(batch_size, drop_remainder=True)
  ds = ds.take(1)
  batch = dataset_utils.tf_to_numpy(next(iter(ds)))
  batch = batch_fn(batch)
  return batch


def take_subset_multi_builder(
    builder: tfds.core.DatasetBuilder, max_examples: int, split_name: str
):
  """Take a subset of each builder composing a multi-builder."""

  def take_subset_split(split):
    if split.num_examples <= max_examples:
      return split
    key = f'{split.name}[:{max_examples}]'
    return tfds.core.splits.SplitDict([split])[key]

  split_dict = builder.info.splits
  splits = []
  for key, split in split_dict.items():
    if key == split_name:
      infos = [take_subset_split(s) for s in split.split_infos]
      split = tfds.core.splits.MultiSplitInfo(name=key, split_infos=infos)
    splits.append(split)
  builder.info.set_splits(tfds.core.splits.SplitDict(splits))


def get_multi_builder(
    config: ml_collections.ConfigDict,
    location_str: str,
    max_examples_per_builder: None | int = None,
) -> tfds.core.DatasetBuilder:
  """Group data from multiple locations into a single builder."""
  locations = sorted(location_str.split(','))
  dirs = []
  all_versions = []
  for location in locations:
    config_dir = os.path.join(config.data_dir, config.dirname, location)
    version = config.version
    if version is None:
      versions = tfds.core.utils.version.list_all_versions(
          os.fspath(config_dir)
      )
      if not versions:
        raise ValueError(
            f'Could not find any version for location {location} at'
            f' {config_dir}.'
        )
      version = versions[-1]
    all_versions.append(version)
    dir_ = os.path.join(config_dir, str(version))
    if config.version is not None and not os.path.isdir(dir_):
      raise ValueError(f'No dataset with version {version} in {config_dir}.')
    dirs.append(dir_)
  if len(dirs) > 1 and not all(v == all_versions[0] for v in all_versions):
    loc_to_version = zip(locations, all_versions)
    raise ValueError(
        f'Not all locations have the same version:\n{loc_to_version}'
    )
  logging.info('Loading location %s at version %s.', location_str, version)
  builder = tfds.core.read_only_builder.builder_from_directories(dirs)
  if max_examples_per_builder is not None:
    take_subset_multi_builder(builder, max_examples_per_builder, 'train')
  return builder


def dataset_iterator_from_builder(
    builder: tfds.core.DatasetBuilder,
    batch_size: int,
    num_shards: int,
    is_training: bool = True,
    split_name: str = 'train',
    process_example_fn: Optional[Callable[[DataDict], DataDict]] = None,
    process_batch_fn: Optional[Callable[[DataDict], DataDict]] = None,
    limit_size: Optional[int] = None,
    shuffle_seed: int = 0,
    shuffle_buffer_size: Optional[int] = None,
    prefetch_buffer_size: Optional[int] = 2,
    threadpool_size: Optional[int] = None,
    read_config: Optional[dict[str, int]] = None,
) -> Iterator[DataDict]:
  """Create the data iterator and returns the builder."""
  shuffle_buffer_size = shuffle_buffer_size or (8 * batch_size)

  # Each host is responsible for a fixed subset of data.
  data_range = tfds.even_splits(split_name, num_shards)[jax.process_index()]
  ds = builder.as_dataset(
      split=data_range,
      shuffle_files=True,
      read_config=tfds.ReadConfig(
          shuffle_seed=shuffle_seed, **(read_config or {})
      ),
  )
  options = tf.data.Options()
  if threadpool_size is not None:
    options.threading.private_threadpool_size = threadpool_size
  ds = ds.with_options(options)

  # Select the subset after shuffling the files to ensure that all locations
  # appear in the subset.
  if limit_size is not None:
    ds = ds.take(limit_size)
  if process_example_fn is not None:
    ds = ds.map(
        process_example_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
  if is_training:
    # First repeat then batch.
    ds = ds.repeat()
    ds = ds.shuffle(shuffle_buffer_size, seed=shuffle_seed)
    ds = ds.batch(batch_size, drop_remainder=True)
  else:
    # First batch then repeat.
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.repeat()
  ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

  maybe_pad_batches = functools.partial(
      dataset_utils.maybe_pad_batch,
      inputs_key=None,
      train=is_training,
      batch_size=batch_size,
      pixel_level=False,
  )
  shard_batches = functools.partial(dataset_utils.shard, n_devices=num_shards)

  ds_iter = iter(ds)
  ds_iter = map(dataset_utils.tf_to_numpy, ds_iter)
  # TODO: do we need padding if drop_remainder=True ?
  ds_iter = map(maybe_pad_batches, ds_iter)
  ds_iter = map(shard_batches, ds_iter)
  if process_batch_fn is not None:
    ds_iter = map(process_batch_fn, ds_iter)
  if prefetch_buffer_size is not None:
    ds_iter = prefetch_to_device_with_strings(ds_iter, prefetch_buffer_size)

  return ds_iter


@datasets.add_dataset(default_configs.streetview_singlescene().name)
def get_dataset(
    *,
    batch_size: int,
    eval_batch_size: int,
    num_shards: int,
    dataset_configs: ml_collections.ConfigDict,
    dtype_str: str = 'float32',
    shuffle_seed: int = 0,
    rng: Optional[jnp.ndarray] = None,
    dataset_service_address: Optional[str] = None,
) -> dataset_utils.Dataset:
  """Returns generators for the train and validation sets."""
  del rng  # the data loading is deterministic
  assert dataset_service_address is None
  assert dataset_configs is not None
  dtype = getattr(tf, dtype_str)

  logging.info('Loading train split.')
  train_location = dataset_configs.locations.training
  assert train_location is not None
  train_builder = get_multi_builder(
      dataset_configs, train_location, dataset_configs.training_size_per_builder
  )

  build_config = types.ProcessingConfig.from_dict(
      train_builder.info.metadata['config']
  )
  if dataset_configs.add_lidar_rays:
    if dataset_configs.num_rays is None:
      dataset_configs.num_rays = build_config.lidar_config.num_rays
  if dataset_configs.pair_scenes:
    dataset_configs.mode = types.DataMode.PAIR_SCENES.value
  if dataset_configs.mode is None:
    dataset_configs.mode = build_config.mode.value
  elif dataset_configs.mode != build_config.mode:
    raise ValueError(
        'The loader and builder expect different data modes:'
        f' {dataset_configs.mode} vs {build_config.mode}'
    )

  # TODO: maybe create these in a base dataset?
  example_fn = functools.partial(
      process_example, dtype=dtype, config=dataset_configs
  )
  batch_fn = functools.partial(process_batch, config=dataset_configs)
  pipeline_options = dict(dataset_configs.get('pipeline_options', tuple()))

  if 'train' in train_builder.info.splits:
    train_iter = dataset_iterator_from_builder(
        train_builder,
        batch_size,
        num_shards,
        is_training=True,
        process_example_fn=example_fn,
        process_batch_fn=batch_fn,
        shuffle_seed=shuffle_seed,
        **pipeline_options,
    )
    training_size = train_builder.info.splits['train'].num_examples
  else:
    train_iter = None
    training_size = 0

  logging.info('Loading eval split.')
  # Get the same data as the training set if no alternative location is given,
  # but allow trimming it to a smaller size.
  eval_location = dataset_configs.locations.evaluation or train_location
  if eval_location == train_location:
    eval_builder = train_builder
  else:
    eval_builder = get_multi_builder(dataset_configs, eval_location)
  if (eval_split_name := 'eval') not in eval_builder.info.splits:
    logging.warning('No evaluation split found in %s.', eval_location)
    eval_split_name = 'train'
  eval_iter = dataset_iterator_from_builder(
      eval_builder,
      eval_batch_size,
      num_shards,
      split_name=eval_split_name,
      is_training=False,
      limit_size=dataset_configs.evaluation_size,
      process_example_fn=example_fn,
      process_batch_fn=batch_fn,
      **pipeline_options,
  )
  evaluation_size = eval_builder.info.splits[eval_split_name].num_examples
  evaluation_size = min(
      dataset_configs.get('evaluation_size', evaluation_size), evaluation_size
  )

  dummy_batch_fn = functools.partial(
      get_dummy_batch,
      batch_size=batch_size,
      builder=train_builder,
      example_fn=example_fn,
      batch_fn=batch_fn,
  )

  voxel_size = dataset_configs.voxel_size
  grid_size_meters = build_config.scene_config.grid_size
  assert (
      grid_size_meters
      == eval_builder.info.metadata['config']['scene_config']['grid_size']
  )
  grid = grids.Grid3D.from_extent_meters(grid_size_meters, voxel_size)
  meta_data = {
      'grid': grid,
      'build_config': build_config,
      'grid_size_meters': grid_size_meters,
      'num_train_examples': training_size,
      'num_eval_examples': evaluation_size,
      'get_dummy_batch_fn': dummy_batch_fn,
      'semantic_map_classes': build_config.rasters_config.semantic_classes,
      'semantic_classes_gt': build_config.rasters_config.gt_semantic_classes,
  }
  return dataset_utils.Dataset(train_iter, eval_iter, None, meta_data)
