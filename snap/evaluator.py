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

"""Compute and dump evaluation metrics."""
import functools
import io
import json

from absl import logging
from etils.array_types import FloatArray
from etils.epath import Path
import jax
import jax.numpy as jnp
from ml_collections import config_dict
import numpy as np
from scenic.dataset_lib import dataset_utils
from scenic.google.xm import xm_utils
from scenic.train_lib import pretrain_utils
from scenic.train_lib import train_utils
import tqdm

from snap import models
from snap.configs import defaults as default_configs
from snap.models import base
from snap.utils import configs as config_utils
from snap.utils import geometry
from snap.utils import misc

CITIES_SPLITS = {
    'train': default_configs.DATA_SPLITS_CITIES['train'],
    'test': ['osaka', 'amsterdam', 'mexico', 'melbourne', 'saopaulo', 'seattles'],
}

ResultDict = dict[str, np.ndarray]


def compute_distance_view_to_map(
    m_t_vq: geometry.Transform3D, m_t_vm: geometry.Transform3D
) -> tuple[FloatArray['...'], FloatArray['...']]:
  dr, dt = (m_t_vq.inv @ m_t_vm).magnitude()
  dt_closest = dt.min(-1)
  dr_closest = dr[jnp.argmin(dt, axis=-1)]
  return dr_closest, dt_closest


@jax.vmap
def pack_localization_metrics(
    training_metrics: base.MetricsDict,
    losses: base.LossDict,
    data: base.Batch,
    pred: base.Predictions,
) -> base.MetricsDict:
  """Generate evaluation metrics for one example."""
  m_t_vq = data['T_query2map'] @ data['query']['T_view2scene'][..., 0]
  dr_closest, dt_closest = compute_distance_view_to_map(
      m_t_vq, data['map']['T_view2scene']
  )

  eval_metrics = dict(
      error_max_meter=training_metrics['loc/err_max_position'],
      error_max_deg=training_metrics['loc/err_max_rotation'],
      recall_top1=training_metrics['loc/recall_top1'],
      pose_score_max=pred['scores_poses'][..., 1:].max(-1),
      overlap=data['overlap'],
      time_delta_days=data['time_delta_days'],
      closest_map_view_meter=dt_closest,
      closest_map_view_deg=dr_closest,
      loss=losses['total'],
  )
  return eval_metrics


def eval_step(
    model_state: train_utils.TrainState,
    batch: base.Batch,
    *,
    rng: jnp.ndarray,
    model: base.BaseModel,
) -> base.MetricsDict:
  """Evaluate a batch of examples."""
  variables = {'params': model_state.params, **model_state.model_state}
  pred = model.flax_model.apply(
      variables,
      batch,
      train=False,
      mutable=False,
      debug=False,
      rngs={'sampling': rng},
  )
  losses, metrics = model.loss_metrics_function(pred, batch, model_state.params)
  match (model_name := model.__class__.__name__):
    case 'BEVLocalizerModel':
      metrics = pack_localization_metrics(metrics, losses, batch, pred)
    case 'SemanticNetModel':
      metrics = model.pack_evaluation_metrics(metrics, losses, batch, pred)
    case _:
      raise ValueError(f'No packing function for model {model_name}.')
  return metrics


def eval_on_dataset(
    model: base.BaseModel,
    state: train_utils.TrainState,
    dataset: dataset_utils.Dataset,
    rng: jnp.ndarray,
    batch_size: int,
) -> base.MetricsDict:
  """Compute and aggregates the metrics for the evaluation set."""
  eval_step_ = jax.jit(functools.partial(eval_step, rng=rng, model=model))
  results = []
  num_examples = dataset.meta_data['num_eval_examples']
  num_steps = int(np.ceil(num_examples / batch_size))
  pbar = tqdm.tqdm(total=num_examples)
  for _ in range(num_steps):
    batch = next(dataset.valid_iter)
    batch = jax.tree_map(lambda x: x.squeeze(0), batch)  # remove shard
    metrics = eval_step_(state, misc.filter_batch_for_jit(batch))
    metrics['vehicle_map'] = batch.get('map', batch)['vehicle_type']
    if 'query' in batch:
      metrics |= dict(
          vehicle_query=batch['query']['vehicle_type'],
          pair_id=batch['pair_id'],
      )
    for i in range(batch_size):
      if not batch['batch_mask'][i]:
        continue
      results.append(jax.tree_map(lambda x: x[i].item(), metrics))  # pylint: disable=cell-var-from-loop
    pbar.update(int(batch['batch_mask'].sum()))
    del batch, metrics
  pbar.close()
  return jax.tree_map(lambda *t: np.array(t), *results)


def get_model_and_dataset(
    eval_config: config_dict.ConfigDict,
    config: config_dict.ConfigDict,
    workdir: Path,
    location: str,
) -> tuple[
    base.BaseModel,
    train_utils.TrainState,
    dataset_utils.Dataset,
    config_dict.ConfigDict,
]:
  """Loads the dataset and model for a given location."""
  config.batch_size = eval_config.batch_size

  # Update the data config: evaluation > experiment > default
  config_data_xp = config_utils.configs_merge(
      getattr(default_configs, config.data.name)(), config.data
  )
  config_data_override = {
      k: getattr(config_data_xp, k)
      for k in ('voxel_size', 'add_images', 'add_lidar_rays', 'add_rasters')
  }
  config.data = config_utils.configs_merge(
      eval_config.data.loader, config_dict.ConfigDict(config_data_override)
  )
  config.data.locations.training = config.data.locations.evaluation = location

  dataset = train_utils.get_dataset(
      config,
      jax.random.PRNGKey(eval_config.data.rng_seed),
      dataset_name=config.data.name,
      dataset_configs=config.data,
  )

  dtype = getattr(jnp, eval_config.dtype_str)
  model_class = models.get_model(config.model_name)
  config_model_default = model_class.default_flax_model_config()
  config.model = config_utils.configs_merge(config_model_default, config.model)
  config.model = config_utils.configs_merge(config.model, eval_config.model)
  model = model_class(config.model, dataset.meta_data, dtype)

  state = pretrain_utils.restore_pretrained_checkpoint(
      str(workdir), assert_exist=True, step=eval_config.checkpoint_step
  )

  device = jax.devices('gpu')[0]
  variables = {'state': state.model_state, 'params': state.params}
  variables = jax.device_put(
      jax.tree_map(lambda x: x.astype(model.dtype), variables), device
  )
  state = state.replace(
      params=variables['params'],
      model_state=variables['state'],
  )

  exper = workdir.parent.name
  logging.info('Loaded experiment %s at step %d.', exper, state.global_step)
  return model, state, dataset, config


def write_eval_dump(
    eval_dir: Path,
    results: dict[str, np.ndarray],
    config: config_dict.ConfigDict,
    compressed: bool = False,
):
  """Write to an evaluation directory."""
  eval_dir.mkdir(parents=True, exist_ok=True)
  io_buffer = io.BytesIO()
  if compressed:
    np.savez_compressed(io_buffer, **results)
  else:
    np.savez(io_buffer, **results)
  (eval_dir / 'results.npz').write_bytes(io_buffer.getvalue())
  config_utils.config_save(eval_dir, config)


def read_eval_dump(
    eval_dir: Path,
) -> tuple[dict[str, np.ndarray], config_dict.ConfigDict]:
  """Read from an evaluation directory."""
  results = eval_dir / 'results.npz'
  results = dict(np.load(io.BytesIO(results.read_bytes()), allow_pickle=False))
  config = config_utils.config_load(eval_dir)
  return results, config


def compute_recall(
    errors: FloatArray['N'], max_error: float
) -> tuple[FloatArray['N'], FloatArray['N']]:
  """Compute the cumulative recall curve."""
  thresholds = np.linspace(0, max_error, 100)
  recall = np.mean(errors < thresholds[:, None], axis=1)
  return thresholds, recall * 100


def run_for_location(
    location: str,
    eval_config: config_dict.ConfigDict,
    fail_if_missing: bool = False,
) -> tuple[ResultDict, config_dict.ConfigDict]:
  """Run the evaluation on one location."""
  workdir = Path(eval_config.workdir)
  experiment_config = config_utils.config_load(workdir)
  eval_path = workdir / 'evaluation' / f'{location}{eval_config.tag}'
  if eval_path.exists() and not eval_config.overwrite:
    logging.info(
        'Loading dump for experiment %d from path %s.',
        workdir,
        eval_path,
    )
    results, config = read_eval_dump(eval_path)
  elif fail_if_missing:
    raise ValueError(
        f'Missing dump for experiment {workdir}, '
        f'expected at path {eval_path}'
    )
  else:
    model, state, dataset, config = get_model_and_dataset(
        eval_config, experiment_config, workdir, location
    )
    results = eval_on_dataset(
        model,
        state,
        dataset,
        jax.random.PRNGKey(eval_config.rng_seed),
        eval_config.batch_size,
    )
    write_eval_dump(eval_path, results, config)
  return results, config


def run(
    config: config_dict.ConfigDict, **kwargs
) -> dict[str, tuple[ResultDict, config_dict.ConfigDict]]:
  """Evaluate on multiple locations sequentially."""
  if jax.device_count() > 1:
    raise ValueError('Only 1 accelerator is supported for now.')
  split = config.data.split
  if split is None:
    raise ValueError('Split is required but is None.')
  cities = CITIES_SPLITS.get(split, split.split(','))
  logging.info('Running evaluation for cities %s.', cities)
  results = {}
  for city in cities:
    location = config.data.name_pattern.format(city)
    logging.info('Running evaluation for location %s.', location)
    results[city] = run_for_location(location, config, **kwargs)
  return results
