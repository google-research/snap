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

"""Utilities for estimating poses from BEV correspondences."""
import math
from typing import Tuple

from etils.array_types import BoolArray
from etils.array_types import FloatArray
from etils.array_types import IntArray
import jax
import jax.numpy as jnp

from snap.utils import geometry
from snap.utils import grids


def sample_sparse_query_points(
    features: FloatArray['H W D'],
    valid: BoolArray['H W'],
    rng: jnp.ndarray,
    grid: grids.Grid2D,
    num_points: int,
) -> Tuple[
    FloatArray['N D'], BoolArray['N'], FloatArray['N 2'], IntArray['N 2']
]:
  """Sample num_points distinct points from a 2D feature map."""
  uv_all = grid.grid_index().reshape(-1, 2)
  assert num_points <= len(uv_all)
  indices = jax.random.choice(rng, len(uv_all), (num_points,), replace=False)
  uv = uv_all[indices]
  xy = (uv + 0.5) * grid.cell_size  # Half-integer pixel centers.
  features_q = features[tuple(uv.T)]
  valid_q = valid[tuple(uv.T)]
  return features_q, valid_q, xy, uv


def interpolate_score_maps(
    scores: FloatArray['N H W'],
    points: FloatArray['N 2'],
    valid: BoolArray['N H W'],
) -> Tuple[FloatArray['N'], BoolArray['N']]:
  interp_many = jax.vmap(grids.interpolate_nd, in_axes=(0, 0, None))
  scores_interp, valid_interp = interp_many(
      scores[..., None],  # add channel dim
      points[..., None, :],  # a single point per score map
      valid,
  )
  return scores_interp.squeeze((-2, -1)), valid_interp.squeeze(-1)


def pose_scoring(
    j_t_i: geometry.Transform2D,
    scores_points_all: FloatArray['N H W'],
    i_xy_points: FloatArray['N 2'],
    valid_points: BoolArray['N'],
    valid_j: BoolArray['H W'],
    grid: grids.Grid2D,
    mask_out_of_bounds: bool,
) -> FloatArray:
  """Compute a consistency score for a given pose."""
  j_uv_points = (j_t_i @ i_xy_points) / grid.cell_size
  scores_points, valid_j_points = interpolate_score_maps(
      scores_points_all,
      j_uv_points,
      valid_j,
  )
  if mask_out_of_bounds:
    valid_points = valid_points & valid_j_points
  score_pose = jnp.sum(valid_points * scores_points, axis=-1)
  return score_pose


def sample_transforms_random(
    rng: jnp.ndarray, num: int, grid: grids.Grid2D
) -> geometry.Transform2D:
  """Randomly sample num poses uniformly within the grid."""
  rng_angle, rng_t = jax.random.split(rng)
  angle = jax.random.uniform(rng_angle, (num,), maxval=jnp.pi * 2)
  grid_size = jnp.asarray(grid.extent_meters)
  t_max = grid_size * 2 / 3
  translation = jax.random.uniform(rng_t, (num, 2), minval=-t_max, maxval=t_max)
  centeri_t_centerj = geometry.Transform2D.from_radians(angle, translation)
  corner_t_center = geometry.Transform2D.from_radians(0, grid_size / 2)
  i_t_j = corner_t_center @ centeri_t_centerj @ corner_t_center.inv
  return i_t_j


def kabsch_algorithm_2d(
    i_p: FloatArray['N 2'], j_p: FloatArray['N 2']
) -> Tuple[geometry.Transform2D, BoolArray, FloatArray]:
  """Compute the least-squares 2D transform between two sets of points."""
  mu_i = i_p.mean(0)
  mu_j = j_p.mean(0)
  i_p = i_p - mu_i
  j_p = j_p - mu_j

  covariance = jnp.einsum('ji,jk->ik', i_p, j_p)
  u, s, vh = jnp.linalg.svd(covariance)

  sign = jnp.sign(jnp.linalg.det(u @ vh))
  u = u * jnp.r_[1, sign]
  s = s * jnp.r_[1, sign]
  valid = s[1] > 1e-16 * s[0]

  error = jnp.sum(jnp.sum(i_p**2 + j_p**2, axis=1)) - 2 * jnp.sum(s)
  rssd = jnp.sqrt(error.clip(min=0))

  i_r_j = jnp.dot(u, vh)
  i_p_j = mu_i - i_r_j @ mu_j
  i_t_j = geometry.Transform2D.from_R(i_r_j, i_p_j)
  return i_t_j, valid, rssd


def sample_transforms_ransac(
    rng: jnp.ndarray,
    prob_points: FloatArray['N H W'],
    i_xy_p: FloatArray['N 2'],
    num_poses: int,
    num_retries: int,
    grid: grids.Grid2D,
) -> geometry.Transform2D:
  """Randomly sample poses derived from the most confident correspondences."""
  shape = prob_points.shape
  prob_points = prob_points.reshape(-1)
  num_matches = math.prod(shape)
  num_obs = 2  # Number of observations required per pose sample.
  indices = jax.random.choice(  # Sample correspondences.
      rng,
      num_matches,
      shape=(num_poses * num_retries * num_obs,),
      replace=True,
      p=prob_points,
  )
  indices = jnp.stack(jnp.unravel_index(indices, shape), -1)
  pool_shape = (num_poses, num_retries, num_obs, 2)
  i_xy_pool = i_xy_p[indices[..., 0]].reshape(pool_shape)
  j_xy_pool = grid.index_to_xyz(indices[..., 1:]).reshape(pool_shape)

  # We sample multiple minimal sets and retain those that are most consistent.
  if num_retries > 1:
    d_i = jnp.linalg.norm(jnp.diff(i_xy_pool, axis=-2).squeeze(-2), axis=-1)
    d_j = jnp.linalg.norm(jnp.diff(j_xy_pool, axis=-2).squeeze(-2), axis=-1)
    ratio = jnp.maximum(d_i / d_j.clip(min=1e-5), d_j / d_i.clip(min=1e-5))
    select_indices = jnp.argmin(ratio, axis=-1)
    select_fn = jax.vmap(lambda x, i: x[i])
    i_xy_pool = select_fn(i_xy_pool, select_indices)
    j_xy_pool = select_fn(j_xy_pool, select_indices)
  else:
    i_xy_pool = i_xy_pool.squeeze(1)
    j_xy_pool = j_xy_pool.squeeze(1)

  j_t_i, _, _ = jax.vmap(kabsch_algorithm_2d)(j_xy_pool, i_xy_pool)
  return j_t_i


def grid_refinement(
    j_t_i_init: geometry.Transform2D,
    scores_points_all: FloatArray['N H W'],
    i_xy_points: FloatArray['N 2'],
    valid_points: BoolArray['N'],
    valid_j: BoolArray['H W'],
    grid: grids.Grid2D,
    mask_out_of_bounds: bool,
) -> tuple[geometry.Transform2D, FloatArray['R H W']]:
  """Scores poses distributed on a regular grid centered at an initial pose."""
  delta_p = 0.2
  delta_r = 0.25
  range_p = 4
  range_r = 5
  slice_p = slice(-range_p, range_p + delta_p, delta_p)
  slice_r = slice(-range_r, range_r + delta_r, delta_r)
  offsets_rxy = jnp.mgrid[slice_r, slice_p, slice_p]

  exhaustive_shape = offsets_rxy.shape[1:]
  offsets_rxy = offsets_rxy.reshape(3, -1).T
  i_t_i_offset = geometry.Transform2D.from_radians(
      angle=jnp.deg2rad(offsets_rxy[..., 0]), t=offsets_rxy[..., 1:]
  )
  j_t_i_samples = j_t_i_init @ i_t_i_offset

  scores = pose_scoring_many(
      j_t_i_samples,
      scores_points_all,
      i_xy_points,
      valid_points,
      valid_j,
      grid,
      mask_out_of_bounds,
  )
  idx_best = jnp.argmax(scores)
  j_t_i_refined = j_t_i_samples[idx_best]
  scores = scores.reshape(exhaustive_shape)
  return j_t_i_refined, scores


pose_scoring_many = jax.vmap(pose_scoring, in_axes=(0,) + (None,) * 6)
pose_scoring_many_batched = jax.vmap(
    pose_scoring_many, in_axes=(0,) * 5 + (None,) * 2
)
grid_refinement_batched = jax.vmap(
    grid_refinement, in_axes=(0,) * 5 + (None,) * 2
)
sample_transforms_random_batched = jax.vmap(
    sample_transforms_random, in_axes=(0,) + (None,) * 2
)
sample_transforms_ransac_batched = jax.vmap(
    sample_transforms_ransac, in_axes=(0,) * 3 + (None,) * 3
)
sample_sparse_query_points_batched = jax.vmap(
    sample_sparse_query_points, in_axes=(0, 0, 0, None, None)
)
interpolate_score_maps_batched = jax.vmap(
    interpolate_score_maps, in_axes=(0, 0, 0)
)
