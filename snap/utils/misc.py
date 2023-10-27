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

"""Various convenience functions."""

from typing import Any, Callable, Iterable
# TODO: look into using jaxtyping for type annotation and checking.

import flax
import jax
import jax.numpy as jnp
import numpy as np


def tree_combine(
    op: Callable[[Iterable[Any]], Any], trees: Iterable[Any]
) -> Any:
  return jax.tree_map(lambda *t: op(t), *trees)


def tree_stack(trees: Iterable[Any]) -> Any:
  return tree_combine(jnp.stack, trees)  # pytype: disable=wrong-arg-types  # jnp-type


def tree_index(tree: Any, i: int) -> Any:
  return jax.tree_map(lambda x: x[i], tree)


def filter_dict(
    d: dict[str, Any], filter_fn: Callable[[str, Any], bool]
) -> dict[str, Any]:
  """Filter out elements of a nested dictionary."""
  d = flax.traverse_util.flatten_dict(d)
  d = filter(lambda kv: filter_fn(*kv), d.items())
  return flax.traverse_util.unflatten_dict(dict(d))


def filter_batch_for_jit(batch: dict[str, Any]) -> dict[str, Any]:
  """Exclude string arrays from a batch nested dictionary."""

  def is_string_array(x: Any) -> bool:
    return isinstance(x, np.ndarray) and x.dtype.type is np.str_

  return filter_dict(batch, lambda _, v: not is_string_array(v))


def find_nested_dict(
    tree: dict[str, Any], target_key: str
) -> None | dict[str, Any]:
  for k, v in tree.items():
    if isinstance(v, dict):
      if k == target_key:
        return v
      ret = find_nested_dict(v, target_key)
      if ret is not None:
        return ret
