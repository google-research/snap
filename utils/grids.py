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

"""3D grid representations."""

import dataclasses
import functools
from typing import Optional, Tuple, TypeVar, Type
from etils.array_types import BoolArray
from etils.array_types import FloatArray
from etils.array_types import IntArray
import jax
import jax.numpy as jnp
import numpy as np

Point = FloatArray['... N']
Index = IntArray['... N']
ID = IntArray['...']
AnyGrid = TypeVar('AnyGrid', bound='GridND')


@dataclasses.dataclass(frozen=True)
class GridND:
  """N-dimensional regular grid.

  Attributes:
    extent: The number of cells along each dimension.
    cell_size: The physical size of each cell, in meters.
    num_cells: The total number of cells.
    extent_meters: The physical size of the grid, in meters.
  """

  extent: Tuple[int, ...]
  cell_size: float

  @classmethod
  def from_extent_meters(
      cls: Type[AnyGrid], extent_meters: Tuple[float, ...], cell_size: float
  ) -> AnyGrid:
    extent = tuple(i / cell_size for i in extent_meters)
    if not all(e % 1 == 0 for e in extent):
      raise ValueError(
          f'The metric grid extent {extent_meters} is not divisible '
          f'by the cell size {cell_size}.'
      )
    return cls(tuple(map(int, extent)), cell_size)

  def xyz_to_index(self, xyz: Point) -> Index:
    return jnp.floor(xyz / self.cell_size).astype(int)

  def index_to_xyz(self, idx: Index) -> Point:
    return (idx + 0.5) * self.cell_size

  def index_to_id(self, idx: Index) -> ID:
    idx = jnp.moveaxis(idx, -1, 0)
    # We can't use model='raise' or out-of-bounds asserts inside jit.
    return jnp.ravel_multi_index(idx, self.extent, mode='clip')

  def id_to_index(self, ids: ID) -> Index:
    return jnp.stack(jnp.unravel_index(ids, self.extent), -1)

  @property
  def num_cells(self) -> int:
    return np.prod(self.extent)

  @property
  def extent_meters(self) -> FloatArray['N']:
    return np.asarray(self.extent) * self.cell_size

  def index_in_grid(self, idx: Index) -> BoolArray['...']:
    return ((idx >= 0) & (idx < np.asarray(self.extent))).all(-1)

  def xyz_in_grid(self, xyz: Point) -> BoolArray['...']:
    return ((xyz >= 0) & (xyz < self.extent_meters)).all(-1)

  def grid_index(self) -> IntArray['i j k N']:
    grid = jnp.mgrid[tuple(slice(None, e) for e in self.extent)]
    return jnp.moveaxis(grid, 0, -1)


@dataclasses.dataclass(frozen=True)
class Grid2D(GridND):
  """2-dimensional regular grid."""

  extent: Tuple[int, int]


@dataclasses.dataclass(frozen=True)
class Grid3D(GridND):
  """3-dimensional regular grid."""

  extent: Tuple[int, int, int]

  def bev(self) -> Grid2D:
    return Grid2D(self.extent[:2], self.cell_size)


map_coordinates_with_channels = jax.vmap(
    jax.scipy.ndimage.map_coordinates,
    in_axes=(-1, None, None, None),
    out_axes=-1,
)


@functools.partial(jax.jit, static_argnames=['mode', 'order'])
def interpolate_nd(
    array: FloatArray['... D'],
    points: FloatArray['K N'],
    valid_array: Optional[BoolArray['...']] = None,
    order: int = 1,
    mode: str = 'nearest',
) -> Tuple[FloatArray['K D'], BoolArray['K']]:
  """Interpolate an N-dimensional array at the given points."""
  size = jnp.asarray(array.shape[:-1])
  valid = jnp.all((points >= 0) & (points < size), -1)
  # The origin of the input is the corner of element (0, 0, 0)
  # but the origin of the indexing is its center.
  points = jnp.moveaxis(points - 0.5, -1, 0)
  values = map_coordinates_with_channels(array, points, order, mode)
  if valid_array is not None:
    nan_mask = jnp.where(valid_array, 0, np.nan)
    nan_points_mask = jax.scipy.ndimage.map_coordinates(
        nan_mask, points, order, mode
    )
    valid &= ~jnp.isnan(nan_points_mask)
  return values, valid


def argmax_nd(scores: FloatArray, grid: GridND) -> Index:
  """Returns the index of the maximum value in an N-dimensional tensor."""
  n = len(grid.extent)
  scores = scores.reshape(*scores.shape[:-n], -1)
  i = jnp.argmax(scores, axis=-1)
  return grid.id_to_index(i)


def expectation_nd(pdf: FloatArray, grid: GridND) -> FloatArray:
  """Returns the index of the expected value of a N-dim probability tensor."""
  n = len(grid.extent)
  reduce_axes = tuple(-i - 2 for i in range(n))
  i = jnp.sum(grid.grid_index() * pdf[..., None], axis=reduce_axes)
  return i
