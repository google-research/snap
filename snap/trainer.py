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

"""Training Script."""

import functools
from typing import Any, Callable, Dict, Optional, Tuple, Type

from absl import logging
from clu import metric_writers
from clu import periodic_actions
from clu import platform
import flax
from flax import jax_utils
from flax.core.scope import FrozenVariableDict
import flax.linen as nn
from flax.training import dynamic_scale as dynamic_scale_lib
import jax
from jax.example_libraries.optimizers import clip_grads
import jax.numpy as jnp
import jax.profiler
import ml_collections
import numpy as np
import optax
from scenic.common_lib import debug_utils
from scenic.dataset_lib import dataset_utils
from scenic.model_lib.base_models import model_utils
from scenic.train_lib import lr_schedules
from scenic.train_lib import optimizers
from scenic.train_lib import train_utils

from snap.models import base
from snap.utils import misc

# Aliases for custom types:
LrFn = Callable[[jnp.ndarray], jnp.ndarray]
PyTree = train_utils.PyTree
AggregatedMetricsDict = Dict[str, Tuple[float, int]]


@flax.struct.dataclass
class TrainState(train_utils.TrainState):
  dynamic_scale: Optional[dynamic_scale_lib.DynamicScale] = None


def reduce_metrics(
    metrics: base.MetricsDict, mask: jnp.ndarray
) -> AggregatedMetricsDict:
  """Aggregates the metrics across examples and accelerators."""
  aggregated = {}
  for k, v in metrics.items():
    metric_mask = mask * jnp.isfinite(v)
    num_examples = metric_mask.sum()
    metric_tuple = (jnp.sum(v, where=metric_mask), num_examples)
    aggregated[k] = model_utils.psum_metric_normalizer(metric_tuple)
  return aggregated  # pytype: disable=bad-return-type  # jax-ndarray


def _gather_variables_recursive(
    m: nn.Module, method_name: str
) -> Dict[str, Any]:
  """Recursively gather variables returned by `method_name` in submodules."""
  if hasattr(m, method_name):
    ret = getattr(m, method_name)()
    if ret is not None:
      return ret
  ret = {}
  for name, child in m._state.children.items():  # pylint: disable=protected-access
    if isinstance(child, str):  # single parameter
      continue
    child_vars = _gather_variables_recursive(child, method_name)
    for k, var in child_vars.items():  # params, stats, etc.
      if k not in ret:
        ret[k] = {}
      ret[k][name] = var
  return ret


def update_pretrained_variables(
    model_def: nn.Module, variables: FrozenVariableDict
) -> FrozenVariableDict:
  """Update the variables from pretrained weights gathered from submodules."""
  pretrained = model_def.apply(
      variables, 'load_pretrained_variables', method=_gather_variables_recursive
  )
  pretrained = flax.traverse_util.flatten_dict(pretrained)
  if not pretrained:
    return variables

  variables = flax.traverse_util.flatten_dict(variables)
  keys_unused = pretrained.keys() - variables.keys()
  keys_update = pretrained.keys() & variables.keys()
  if keys_unused:
    logging.info(
        'The following pretrained variables will not be used:\n%s',
        '\n'.join(map('.'.join, sorted(keys_unused))),
    )
    if not keys_update:
      raise ValueError(
          'Could not load any pre-trained weight, all were left unused.'
      )
  logging.info(
      'Updating %d variable(s) from pretrained weights.', len(keys_update)
  )
  for k in keys_update:
    variables[k] = pretrained[k].astype(variables[k].dtype)
  variables = flax.traverse_util.unflatten_dict(variables)
  variables = flax.core.frozen_dict.freeze(variables)
  return variables


def initialize_model(
    *,
    model_def: nn.Module,
    dummy_input: base.Batch,
    rng: jnp.ndarray,
) -> Tuple[PyTree, PyTree, int]:
  """Initializes parameters and model state.

  Differently from train_utils.initialize_model, expects a dummy batch as input
  instead of building one from (shape, dtype) specs. This allows us to use batch
  dictionaries with arbitrary objects, not only jax arrays.

  Args:
    model_def: Definition of a model.
    dummy_input: Batch dictionary.
    rng: Jax rng key.

  Returns:
    Initial params, Init model_state, and number of trainable_params.
  """

  # We want all parameters to be created in host RAM, not on any device, they'll
  # be sent there later as needed, otherwise we already encountered two
  # situations where we allocate them twice.
  @functools.partial(jax.jit, backend='cpu')
  def _initialize_model(rngs):
    """Initialization function to be jitted."""
    variables = model_def.init(rngs, dummy_input, train=False, debug=False)
    variables = update_pretrained_variables(model_def, variables)
    init_model_state, init_params = variables.pop('params')
    return init_params, init_model_state

  rngs = dict(zip(('params', 'sampling'), jax.random.split(rng, 2)))
  init_params, init_model_state = _initialize_model(rngs)
  # Pop out params rng:
  rngs.pop('params')

  # Count number of trainable parameters:
  num_trainable_params = debug_utils.log_param_shapes(init_params)
  return init_params, init_model_state, num_trainable_params


def train_step(
    train_state: TrainState,
    batch: base.Batch,
    *,
    flax_model: nn.Module,
    loss_metrics_fn: base.LossMetricsFn,
    lr_fn: LrFn,
    config: ml_collections.ConfigDict,
    debug: Optional[bool] = False,
) -> Tuple[TrainState, AggregatedMetricsDict, Dict[str, Any]]:
  """Runs a single step of training.

  Given the state of the training and a batch of data, computes
  the loss and updates the parameters of the model.

  Note that in this code, the buffers of the first (train_state) and second
  (batch) arguments are donated to the computation.

  Args:
    train_state: The state of training including the current global_step,
      model_state, rng, params, and optimizer. The buffer of this argument can
      be donated to the computation.
    batch: A single batch of data. The buffer of this argument can be donated to
      the computation.
    flax_model: A Flax model.
    loss_metrics_fn: A function that, given predictions, a batch, and parameters
      of the model, calculates the loss and metrics dictionaries.
    lr_fn: The learning rate fn used for the logging the learning rate.
    config: Configurations of the experiment.
    debug: Whether the debug mode is enabled during training. `debug=True`
      enables model specific logging/storing some values using
      jax.host_callback.

  Returns:
    Updated state of training and computed metrics and some training logs.
  """
  training_logs = {}
  new_rng, rng = jax.random.split(train_state.rng)

  # Bind the rng to the host/device we are on.
  sampling_rng = train_utils.bind_rng_to_host_device(
      rng, axis_name='batch', bind_to='device'
  )

  def training_loss_fn(params):
    variables = {'params': params, **train_state.model_state}
    pred, new_model_state = flax_model.apply(
        variables,
        batch,
        mutable=['batch_stats'],
        train=True,
        rngs={'sampling': sampling_rng},
        debug=debug,
    )
    losses, metrics = loss_metrics_fn(pred, batch, params)
    loss = losses['total'].mean(where=batch['batch_mask'])
    return loss, (new_model_state, losses, metrics)

  dynamic_scale = train_state.dynamic_scale
  if dynamic_scale:
    grad_fn = dynamic_scale.value_and_grad(
        training_loss_fn, has_aux=True, axis_name='batch'
    )
    dynamic_scale, is_fin, (_, aux), grad = grad_fn(train_state.params)
    # dynamic loss takes care of averaging gradients across replicas
  else:
    grad_fn = jax.grad(training_loss_fn, has_aux=True)
    grad, aux = grad_fn(train_state.params)
    # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
    grad = jax.lax.pmean(grad, axis_name='batch')

  new_model_state, losses, metrics = aux

  if config.get('max_grad_norm') is not None:
    grad = clip_grads(grad, config.max_grad_norm)

  assert train_state.tx is not None
  updates, new_opt_state = train_state.tx.update(
      grad, train_state.opt_state, train_state.params
  )
  new_params = optax.apply_updates(train_state.params, updates)

  training_logs['l2_grads'] = jnp.sqrt(
      sum([jnp.vdot(g, g) for g in jax.tree_util.tree_leaves(grad)])
  )
  us = jax.tree_util.tree_leaves(updates)
  training_logs['l2_updates'] = jnp.sqrt(sum([jnp.vdot(u, u) for u in us]))
  # Can we get this from the optimizer instead?
  global_step = jnp.array([train_state.global_step])
  training_logs['learning_rate'] = lr_fn(global_step)

  for k, v in losses.items():
    metrics[f'loss/{k}'] = v
  metrics = reduce_metrics(metrics, batch['batch_mask'])

  if not dynamic_scale:
    is_fin = [
        jax.lax.is_finite(g).all() for g in jax.tree_util.tree_leaves(grad)
    ]
    is_fin = jnp.all(jnp.stack(is_fin))
  else:
    metrics['scale'] = (dynamic_scale.scale, 1)
  # if is_fin == False the gradients contain Inf/NaNs and optimizer state and
  # params should be restored (= skip this step).
  new_opt_state = jax.tree_util.tree_map(
      functools.partial(jnp.where, is_fin),
      new_opt_state,
      train_state.opt_state,
  )
  new_params = jax.tree_util.tree_map(
      functools.partial(jnp.where, is_fin), new_params, train_state.params
  )
  training_logs['is_finite'] = is_fin

  # Plot l2 norm of params after we got rid of inf and NaN.
  ps = jax.tree_util.tree_leaves(new_params)
  # Avoid float16 overflows.
  training_logs['l2_params'] = jnp.sqrt(
      sum([jnp.vdot(p.astype(jnp.float32), p.astype(jnp.float32)) for p in ps])
  )

  new_train_state = train_state.replace(  # pytype: disable=attribute-error
      global_step=train_state.global_step + 1,
      opt_state=new_opt_state,
      params=new_params,
      model_state=new_model_state,
      rng=new_rng,
      dynamic_scale=dynamic_scale,
  )

  return new_train_state, metrics, training_logs


def eval_step(
    train_state: TrainState,
    batch: base.Batch,
    rng: jnp.ndarray,
    *,
    flax_model: nn.Module,
    loss_metrics_fn: base.LossMetricsFn,
    debug: Optional[bool] = False,
) -> Tuple[AggregatedMetricsDict, base.Predictions]:
  """Runs a single step of training.

  Note that in this code, the buffer of the second argument (batch) is donated
  to the computation.

  Assumed API of metrics_fn is:
  ```metrics = metrics_fn(pred, batch)
  where batch is yielded by the batch iterator, and metrics is a dictionary
  mapping metric name to a vector of per example measurements. eval_step will
  aggregate (by summing) all per example measurements and divide by the
  aggregated normalizers. For each given metric we compute:
  1/N sum_{b in batch_iter} metric(b), where  N is the sum of normalizer
  over all batches.

  Args:
    train_state: TrainState, the state of training including the current
      global_step, model_state, rng, params and optimizer state. The buffer of
      this argument can be donated to the computation.
    batch: A single batch of data.
    rng: Jax rng key.
    flax_model: A Flax model.
    loss_metrics_fn: A function that, given predictions, a batch, and parameters
      of the model, calculates the loss and metrics dictionaries.
    debug: Whether the debug mode is enabled during evaluation. `debug=True`
      enables model specific logging/storing some values using
      jax.host_callback.

  Returns:
    Calculated metrics and predictions.
  """
  sampling_rng = train_utils.bind_rng_to_host_device(
      rng, axis_name='batch', bind_to='device'
  )
  variables = {'params': train_state.params, **train_state.model_state}
  pred = flax_model.apply(
      variables,
      batch,
      train=False,
      mutable=False,
      debug=debug,
      rngs={'sampling': sampling_rng},
  )
  losses, metrics = loss_metrics_fn(pred, batch, train_state.params)
  for k, v in losses.items():
    metrics[f'loss/{k}'] = v
  metrics = reduce_metrics(metrics, batch['batch_mask'])
  return metrics, pred


def train(
    *,
    rng: jnp.ndarray,
    config: ml_collections.ConfigDict,
    model_cls: Type[base.BaseModel],
    dataset: dataset_utils.Dataset,
    workdir: str,
    writer: metric_writers.MetricWriter,
) -> Tuple[train_utils.TrainState, Dict[str, Any], Dict[str, Any]]:
  """Main training loop lives in this function.

  Given the model class and dataset, it prepares the items needed to run the
  training, including the TrainState.

  Args:
    rng: Jax rng key.
    config: Configurations of the experiment.
    model_cls: Model class; A model has a flax_module, a loss_fn, and a
      metrics_fn associated with it.
    dataset: The dataset that has train_iter, eval_iter, meta_data, and
      optionally, test_iter.
    workdir: Directory for checkpointing.
    writer: CLU metrics writer instance.

  Returns:
    train_state that has the state of training (including current
      global_step, model_state, rng, and the optimizer), train_summary
      and eval_summary which are dict of metrics. These outputs are used for
      regression testing.
  """
  lead_host = jax.process_index() == 0
  dtype = getattr(jnp, config.dtype_str)
  match dtype:
    case jnp.float32 | jnp.bfloat16:
      dynamic_scale = None
    case jnp.float16:
      dynamic_scale = dynamic_scale_lib.DynamicScale(minimum_scale=256)
    case _:
      raise ValueError(f'Unsupported dtype: {config.dtype_str}')

  # Build the loss_fn, metrics, and flax_model.
  model = model_cls(config.model, dataset.meta_data, dtype)

  # Initialize model.
  rng, init_rng = jax.random.split(rng)
  params, model_state, num_trainable_params = initialize_model(
      model_def=model.flax_model,
      dummy_input=dataset.meta_data['get_dummy_batch_fn'](),
      rng=init_rng,
  )

  # Create optimizer.
  lr_fn = lr_schedules.get_learning_rate_fn(config)
  optimizer_config = optimizers.get_optax_optimizer_config(config)
  # If the config is already an optax-compatible config, better call directly:
  #   optimizers.get_optimizer(config.optimizer_configs, lr_fn)
  tx = optimizers.get_optimizer(optimizer_config, lr_fn, params=params)
  # We jit this, such that the arrays that are created on the same device as the
  # input is, in this case the CPU. Else they'd be on device[0].
  opt_state = jax.jit(tx.init, backend='cpu')(params)

  rng, train_rng = jax.random.split(rng)
  rng, eval_rng = jax.random.split(rng)

  # Create chrono class to track and store training statistics and metadata:
  chrono = train_utils.Chrono()

  # False positive
  # pylint: disable=unexpected-keyword-arg
  train_state = TrainState(
      global_step=0,
      opt_state=opt_state,
      tx=tx,
      params=params,
      model_state=model_state,
      rng=train_rng,
      metadata={'chrono': chrono.save()},
      dynamic_scale=dynamic_scale,
  )
  # pylint: enable=unexpected-keyword-arg
  start_step = train_state.global_step
  if config.checkpoint:
    train_state, start_step = train_utils.restore_checkpoint(
        workdir, train_state
    )
  chrono.load(train_state.metadata['chrono'])
  train_state = train_state.replace(metadata={})
  # Replicate the optimizer, state, and rng.
  train_state = jax_utils.replicate(train_state)
  del params  # Do not keep a copy of the initial params.

  # Calculate the total number of training steps.
  total_steps, steps_per_epoch = train_utils.get_num_training_steps(
      config, dataset.meta_data
  )

  train_step_pmapped = jax.pmap(
      functools.partial(
          train_step,
          flax_model=model.flax_model,
          loss_metrics_fn=model.loss_metrics_function,
          lr_fn=lr_fn,
          config=config,
          debug=config.debug_train,
      ),
      axis_name='batch',
      # We can donate both buffers of train_state and train_batch.
      donate_argnums=(0, 1),
  )
  eval_step_pmapped = jax.pmap(
      functools.partial(
          eval_step,
          flax_model=model.flax_model,
          loss_metrics_fn=model.loss_metrics_function,
          debug=config.debug_eval,
      ),
      in_axes=(0, 0, None),
      axis_name='batch',
      # We can donate the eval_batch's buffer.
      donate_argnums=(1,),
  )
  log_eval_steps = config.get('log_eval_steps') or steps_per_epoch
  if not log_eval_steps:
    raise ValueError("'log_eval_steps' should be specified in the config.")
  checkpoint_steps = config.get('checkpoint_steps') or log_eval_steps
  log_summary_steps = config.get('log_summary_steps') or log_eval_steps

  # Ceil rounding such that we include the last incomplete batch.
  eval_batch_size = config.get('eval_batch_size', config.batch_size)
  total_eval_steps = int(
      np.ceil(dataset.meta_data['num_eval_examples'] / eval_batch_size)
  )
  steps_per_eval = config.get('steps_per_eval') or total_eval_steps

  train_metrics, extra_training_logs = [], []
  train_summary, eval_summary = None, None

  chrono.inform(start_step, total_steps, config.batch_size, steps_per_epoch)
  logging.info('Starting training loop at step %d.', start_step + 1)
  report_progress = periodic_actions.ReportProgress(
      num_train_steps=total_steps,
      writer=writer,
      every_secs=None,
      every_steps=config.get('report_progress_step', log_summary_steps),
  )

  def write_note(note):
    if lead_host:
      platform.work_unit().set_notes(note)

  hooks = []
  if lead_host:
    hooks.append(report_progress)
  if config.get('xprof', True) and lead_host:
    hooks.append(periodic_actions.Profile(num_profile_steps=5, logdir=workdir))

  if start_step == 0:
    step0_log = {'num_trainable_params': num_trainable_params}
    writer.write_scalars(1, step0_log)

  write_note(f'First step compilations...\n{chrono.note}')
  for step in range(start_step + 1, total_steps + 1):
    with jax.profiler.StepTraceAnnotation('train', step_num=step):
      train_batch = misc.filter_batch_for_jit(next(dataset.train_iter))
      train_state, t_metrics, t_logs = train_step_pmapped(
          train_state, train_batch
      )
      # This will accumulate metrics in TPU memory up to the point that we log
      # them. This is no problem for small metrics but may be a problem for
      # large (e.g. segmentation) metrics. An alternative is to set
      # `log_summary_steps` to a small number, or to use
      # `train_utils.unreplicate_and_get` here instead of right before writing
      # summaries, but that means in each step, we have data transfer between
      # tpu and host, which might slow down the training.
      train_metrics.append(t_metrics)
      # Additional training logs: learning rate:
      t_logs = jax.tree_util.tree_map(jax_utils.unreplicate, t_logs)
      extra_training_logs.append(t_logs)
    for h in hooks:
      h(step)
    # Below are once-in-a-while ops -> pause.
    ###################### LOG TRAIN SUMMARY ########################
    if (
        (step % log_summary_steps == 1)
        or (step == total_steps)
        # Don't log during the warmup steps after restoring a checkpoint.
        or (lead_host and chrono.warmup and not start_step)
    ):
      chrono.pause(wait_for=(train_metrics))
      if lead_host:
        chrono.tick(step, writer, write_note)
      # train_metrics is list of a dictionaries of metrics, where the shape of
      # the metrics[key] is [n_local_devices]. However, because metric functions
      # have a psum, we have already summed across the whole sharded batch, and
      # what's returned is n_local_devices copies of the same summed metric.
      # So we do unreplicate and fetch them to host using `unreplicate_and_get`.
      extra_training_logs = jax.tree_util.tree_map(
          jax.device_get, extra_training_logs
      )
      extra_training_logs = [
          t for t in extra_training_logs if t.get('is_finite', True)
      ]
      train_summary = train_utils.log_train_summary(
          step=step,
          train_metrics=jax.tree_util.tree_map(
              train_utils.unreplicate_and_get, train_metrics
          ),
          extra_training_logs=extra_training_logs,
          writer=writer,
          key_separator='/',
      )
      # Reset metric accumulation for next evaluation cycle.
      train_metrics, extra_training_logs = [], []
      chrono.resume()
    ################### EVALUATION #######################
    if (step % log_eval_steps == 1) or (step == total_steps):
      chrono.pause(wait_for=(train_state.params))
      with report_progress.timed('eval'):
        eval_metrics = []
        # Sync model state across replicas.
        train_state = train_utils.sync_model_state_across_replicas(train_state)
        for i in range(steps_per_eval):
          eval_batch = misc.filter_batch_for_jit(next(dataset.valid_iter))
          e_metrics, _ = eval_step_pmapped(
              train_state, eval_batch, jax.random.fold_in(eval_rng, i)
          )
          eval_metrics.append(train_utils.unreplicate_and_get(e_metrics))
        eval_summary = train_utils.log_eval_summary(
            step=step,
            eval_metrics=eval_metrics,
            writer=writer,
            prefix='eval',
            key_separator='/',
        )
      writer.flush()
      del eval_metrics
      chrono.resume()
    ##################### CHECKPOINTING ###################
    if (
        (step % checkpoint_steps == 1 and step > 1) or (step == total_steps)
    ) and config.checkpoint:
      chrono.pause(wait_for=(train_state.params, train_state.opt_state))
      with report_progress.timed('checkpoint'):
        train_utils.handle_checkpointing(
            train_state, chrono, workdir, max_checkpoints_to_keep=10
        )
      chrono.resume()
  # Wait until computations are done before exiting.
  train_utils.barrier_across_hosts()
  # Return the train and eval summary after last step for regression testing.
  assert train_state is not None
  assert train_summary is not None
  assert eval_summary is not None
  return train_state, train_summary, eval_summary
