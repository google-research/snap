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

"""Plotting of 2D data from 2D overhead Bird's-Eye Views."""

from typing import List, Optional, Tuple

from etils.array_types import FloatArray
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from snap.utils import geometry
from snap.utils import grids


def remove_plot_axes(ax: mpl.axes.Axes | None = None, color: str | None = None):
  ax = ax or plt.gca()
  ax.get_xaxis().set_ticks([])
  ax.get_yaxis().set_ticks([])
  for spine in ax.spines.values():
    if color is None:
      spine.set_visible(False)
    else:
      spine.set_edgecolor(color)


def rasterize_pointcloud(
    xy: FloatArray['N 2'],
    grid: Optional[grids.Grid2D] = None,
    values: Optional[FloatArray['N']] = None,
    resolution: float = 0.2,
    max_count: int = 10,
) -> Tuple[FloatArray['H W'], List[float]]:
  """2D horizontal rasterization of a 3D point cloud, with count or values."""
  xy = xy[:, :2]  # pick first 2 dimensions if 3D
  if grid is None:
    origin = xy.min(0) - resolution / 2
    xy = xy - origin
    max_ = xy.max(0) + resolution / 2
    size = np.ceil(max_ / resolution).clip(min=1).astype(int)
    grid = grids.Grid2D(tuple(size), resolution)
  else:
    origin = np.zeros(2)
  ij = grid.xyz_to_index(xy)
  valid = grid.xyz_in_grid(xy)
  raster = np.full(grid.extent, np.nan)
  if values is None:
    ij_select, counts = np.unique(ij[valid], axis=0, return_counts=True)
    raster[tuple(ij_select.T)] = 1 - np.clip(counts / max_count, 0, 1)
  else:
    raster[tuple(ij[valid].T)] = values[valid]
  extent = np.stack([origin, origin + grid.extent_meters]).T.reshape(-1)
  return raster, extent


def get_mpl_transform2d(
    world_t_elem: geometry.TransformND,
) -> mpl.transforms.Affine2D:
  """Transform to matplotlib 2D transform."""
  if isinstance(world_t_elem, geometry.Transform3D):
    world_t_elem = geometry.Transform2D.from_Transform3D(world_t_elem)
  angle = np.rad2deg(world_t_elem.angle)
  return mpl.transforms.Affine2D().rotate_deg(angle).translate(*world_t_elem.t)


class Plotter2D:
  """Plot 3D data, like lidar, grids, or cameras, from a 2D overhead view."""

  def __init__(self, ax=None, **kwargs):
    if ax is None:
      _, ax = plt.subplots(1, 1, **kwargs)
    self.ax = ax
    self.ax.set_aspect('equal')

  def plot_grid(
      self,
      grid: grids.GridND,
      w_t_grid: Optional[geometry.TransformND] = None,
      fill: bool = False,
      **kwargs,
  ):
    """Plot a 3D grid as a box."""
    width, height = grid.extent_meters[:2]
    rec = mpl.patches.Rectangle(np.zeros(2), width, height, fill=fill, **kwargs)
    if w_t_grid is not None:
      rec.set_transform(get_mpl_transform2d(w_t_grid) + self.ax.transData)
    self.ax.add_patch(rec)
    self.ax.autoscale_view()

  def plot_camera(
      self,
      w_t_cam: geometry.TransformND,
      fov_degrees: float = 72,
      scale: float = 0.5,
      dot: bool = True,
      **kwargs,
  ):
    """Plot a camera frustum."""
    x = np.tan(np.deg2rad(fov_degrees) / 2)
    corners = np.array([[0, 0], [x, 1], [-x, 1]])
    if dot:
      self.ax.scatter(*w_t_cam.t[:2], **(kwargs | dict(lw=0, s=15)))
    frustum = plt.Polygon(corners * scale, fill=False, **kwargs)
    frustum.set_transform(get_mpl_transform2d(w_t_cam) + self.ax.transData)
    self.ax.add_patch(frustum)
    self.ax.autoscale_view()

  def plot_raster(
      self,
      raster: FloatArray['H W'],
      extent: Optional[List[float]] = None,
      grid: Optional[grids.GridND] = None,
      w_t_raster: Optional[geometry.TransformND] = None,
      scale: bool = False,
      origin: str = 'lower',
      **kwargs,
  ):
    """Plot a rasterized point cloud."""
    if extent is None:
      if grid is None:
        raise ValueError('Provide either extent or grid.')
      extent = np.stack([np.zeros(2), grid.extent_meters[:2]]).T.reshape(-1)
    self.ax.autoscale(enable=scale)
    raster = np.swapaxes(raster, 0, 1)
    im = self.ax.imshow(raster, extent=extent, origin=origin, **kwargs)
    if w_t_raster is not None:
      im.set_transform(get_mpl_transform2d(w_t_raster) + self.ax.transData)
