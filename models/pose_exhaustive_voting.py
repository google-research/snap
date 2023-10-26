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

"""Utilities for pose estimation via exhaustive voting."""
import functools
import math
from typing import Tuple, Optional

from etils.array_types import BoolArray
from etils.array_types import FloatArray
from etils.array_types import IntArray
import jax
import jax.numpy as jnp

from snap.models import types
from snap.utils import geometry
from snap.utils import grids


def get_grid_center_transform(grid: grids.Grid2D) -> geometry.Transform2D:
  """Returns corner_t_center for a grid."""
  center_offset = jnp.asarray(grid.extent_meters) / 2
  return geometry.Transform2D.from_radians(0, center_offset)


def sample_query_templates(
    features: FloatArray['H W D'],
    valid: BoolArray['H W'],
    num_rotations: int,
    grid: grids.Grid2D,
) -> Tuple[FloatArray['R H W D'], BoolArray['R H W']]:
  """Rotate a BEV by num_rotations uniformly distributed angles."""
  # Compute the transform from the canonical grid to each template rotation.
  angles = jnp.linspace(0, jnp.pi * 2, num_rotations, endpoint=False)
  rotated_t_grid = geometry.Transform2D.from_radians(
      angles, jnp.zeros((len(angles), 2))
  )
  corner_t_center = get_grid_center_transform(grid)
  templates_t_grid = corner_t_center @ rotated_t_grid @ corner_t_center.inv

  # For efficiency, we wrap only the first quadrant (quarter of rotations).
  grid_xy = grid.index_to_xyz(grid.grid_index()).reshape(-1, 2)
  templates_xy = templates_t_grid[: num_rotations // 4] @ grid_xy
  templates_uv = templates_xy / grid.cell_size
  interp_templates = jax.vmap(grids.interpolate_nd, in_axes=(None, 0, None))
  quarter, t_valid = interp_templates(features, templates_uv, valid)
  quarter = jnp.where(t_valid[..., None], quarter, 0)
  quarter = quarter.reshape(-1, *grid.extent, quarter.shape[-1])

  # Complete the other quadrants.
  t_valid = t_valid.reshape(-1, *grid.extent)
  templates = jnp.concatenate(
      [jnp.rot90(quarter, k, axes=(2, 1)) for k in range(4)], 0
  )
  t_valid = jnp.concatenate(
      [jnp.rot90(t_valid, k, axes=(2, 1)) for k in range(4)], 0
  )
  return templates, t_valid


def template_matching(
    q: FloatArray['R H W D'],
    q_valid: BoolArray['R H W'],
    m: FloatArray['H W D'],
    m_valid: BoolArray['H W'],
    do_padding: bool = True,
    min_overlap: Optional[float] = 0.05,
) -> FloatArray['R H W']:
  """Exhaustively match a rotated BEV template with a map."""
  map_size = m.shape[:2]
  if do_padding:
    m = jnp.pad(
        m, tuple((s - 1,) * 2 for s in map_size) + ((0, 0),), mode='edge'
    )
  mode = 'valid' if do_padding else 'full'
  fn_conv = functools.partial(jax.scipy.signal.convolve, mode=mode)
  fn_conv_with_channels = jax.vmap(fn_conv, in_axes=-1, out_axes=-1)
  fn_conv_templates = jax.vmap(fn_conv_with_channels, in_axes=(0, None))
  scores = fn_conv_templates(q[:, ::-1, ::-1, :], m)
  scores = scores.sum(-1)

  if min_overlap is not None:
    m_valid = jnp.pad(
        m_valid, tuple((s - 1,) * 2 for s in map_size), mode='constant'
    )
    num_valid = fn_conv_templates(
        q_valid[..., None], m_valid[..., None]
    ).squeeze(-1)
    valid_score = num_valid > (min_overlap * math.prod(q_valid.shape[-2:]))
    scores = jnp.where(valid_score, scores, -jnp.inf)

  scores /= q_valid.sum((-1, -2), keepdims=True)  # uniform
  return scores


@functools.partial(jax.jit, static_argnames=('num_rotations', 'grid'))
def exhaustive_pose_voting(
    plane_q: types.FeaturePlane,
    plane_map: types.FeaturePlane,
    num_rotations: int,
    grid: grids.Grid2D,
    conf_q: Optional[FloatArray['H W']] = None,
) -> FloatArray['R H W']:
  """Vote over the 3D pose volume."""
  feats_q = plane_q.features
  if conf_q is not None:
    feats_q *= conf_q[..., None]
  templates, t_valid = sample_query_templates(
      feats_q, plane_q.valid, num_rotations, grid
  )
  return template_matching(
      templates, t_valid, plane_map.features, plane_map.valid
  )


def exhaustive_index_to_tfm(
    index: IntArray['3'], grid: grids.Grid2D, num_rotations: int
) -> geometry.Transform2D:
  """Convert a pose volume index to a 3-DoF transform."""
  xy_cell = (index[1:] - jnp.array(grid.extent) + 1 + 0.5) * grid.cell_size
  angle = index[0] * 2 * jnp.pi / num_rotations
  m_t_q_center = geometry.Transform2D.from_radians(-angle, xy_cell)
  corner_t_center = get_grid_center_transform(grid)
  m_t_q_corner = corner_t_center @ m_t_q_center @ corner_t_center.inv
  return m_t_q_corner


def exhaustive_tfm_to_index(
    m_t_q_corner: geometry.Transform2D, grid: grids.Grid2D, num_rotations: int
) -> IntArray['3']:
  """Convert a 3-DoF transform to a pose volume index."""
  # We usually express transforms w.r.t to the grid origin (lower corner)
  # but exahustive PDFs are computed w.r.t grid centers of both map and query.
  corner_t_center = get_grid_center_transform(grid)
  m_t_q_center = corner_t_center.inv @ m_t_q_corner @ corner_t_center
  k = (-m_t_q_center.angle / (jnp.pi * 2) % 1) * num_rotations
  ij = (m_t_q_center.t / grid.cell_size) + jnp.array(grid.extent) - 1.5
  return jnp.concatenate([k[..., None], ij], -1)
