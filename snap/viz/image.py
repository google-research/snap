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

"""2D visualizations in image space."""
import io
from typing import Any, List, Optional, Sequence

from etils.array_types import BoolArray
from etils.array_types import FloatArray
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def plot_images(
    imgs: List[np.ndarray],
    titles: Optional[List[str]] = None,
    cmaps: str = 'gray',
    dpi: int = 100,
    pad: float = 0.5,
    adaptive: bool = True,
    fig_edge_size: float = 4,
    **kwargs,
) -> plt.Figure:
  """Plot a list of images."""
  n = len(imgs)
  if not isinstance(cmaps, (list, tuple)):
    cmaps = [cmaps] * n

  if adaptive:
    ratios = [i.shape[1] / i.shape[0] for i in imgs]  # W / H
  else:
    ratios = [4 / 3] * n
  figsize = [sum(ratios) * fig_edge_size, fig_edge_size]
  fig, ax = plt.subplots(
      1, n, figsize=figsize, dpi=dpi, gridspec_kw={'width_ratios': ratios}
  )
  if n == 1:
    ax = [ax]
  for i in range(n):
    ax[i].imshow(imgs[i], cmap=cmaps[i], **kwargs)
    ax[i].get_yaxis().set_ticks([])
    ax[i].get_xaxis().set_ticks([])
    ax[i].set_axis_off()
    for spine in ax[i].spines.values():  # remove frame
      spine.set_visible(False)
    if titles:
      ax[i].set_title(titles[i])
  fig.tight_layout(pad=pad)
  return fig


def add_categorical_legend(
    classes: Sequence[str],
    labels: None | FloatArray['H W'] = None,
    ax: None | mpl.axes.Axes = None,
    cmap: str = 'turbo',
    ncol: int = 10,
    **kwargs,
):
  """Add a categorical legend, for example for semantic segmentation."""
  if labels is None:
    num_labels = len(classes)
  else:  # Useful if vmax was inferred from the maximum value of the labels.
    num_labels = int(labels[np.isfinite(labels)].max()) + 1
  cmap = mpl.cm.get_cmap(cmap)
  handles = [
      mpl.patches.Rectangle((0, 0), 1, 1, color=cmap(i))
      for i in np.linspace(0, 1, num_labels)
  ]
  (ax or plt.gca()).legend(handles, classes, ncol=ncol, **kwargs)


def _normalize(x):
  return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(min=1e-3)


def load_pca_state(state: str | dict[str, Any]) -> PCA:
  """Load a fitted PCA from a file or a state dictionary."""
  if not isinstance(state, dict):  # it's a path
    with open(state, 'rb') as fid:
      state = dict(np.load(io.BytesIO(fid.read()), allow_pickle=True))
  pca = PCA()
  pca.__setstate__(state)
  return pca


def save_pca_state(path: str, pca: PCA):
  """Save a fitted PCA to a file."""
  state = pca.__getstate__()
  io_buffer = io.BytesIO()
  np.savez(io_buffer, **state)
  with open(path, 'wb') as fid:
    fid.write(io_buffer.getvalue())


def features_to_rgb(
    *feature_maps: FloatArray['H W N'],
    masks: Optional[Sequence[BoolArray['H W']]] = None,
    skip: int = 1,
    pca: PCA | None = None,
    return_pca: bool = False,
) -> list[FloatArray['H W 3']] | tuple[list[FloatArray['H W 3']], PCA]:
  """Project a list of d-dimensional feature maps to RGB colors using PCA."""
  features = []
  for i, fmap in enumerate(feature_maps):
    if masks is not None:
      fmap = fmap[masks[i]]
    features.append(fmap.reshape(-1, fmap.shape[-1]))
  features = np.concatenate(features, axis=0)
  features = _normalize(features)

  if pca is not None:
    rgb = pca.transform(features)
  else:
    pca = PCA(n_components=3)
    if skip > 1:
      pca.fit(features[::skip])
      rgb = pca.transform(features)
    else:
      rgb = pca.fit_transform(features)
  rgb = (_normalize(rgb) + 1) / 2

  rgb_maps = []
  for i, fmap in enumerate(feature_maps):
    h, w, *_ = fmap.shape
    if masks is None:
      rgb_map, rgb = np.split(rgb, [h * w], axis=0)
      rgb_map = rgb_map.reshape((h, w, 3))
    else:
      mask = masks[i]
      rgb_masked, rgb = np.split(rgb, [np.count_nonzero(mask)], axis=0)
      rgb_map = np.zeros((h, w, 4))
      rgb_map[mask, :3] = rgb_masked
      rgb_map[mask, -1] = 1
    rgb_maps.append(rgb_map)
  assert rgb.shape[0] == 0
  if return_pca:
    return rgb_maps, pca
  return rgb_maps
