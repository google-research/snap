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

"""jax wrappers for geometry transformations."""

from typing import Any, Dict, Tuple, Union

import dataclass_array as dca
from etils.array_types import BoolArray
from etils.array_types import FloatArray
import jax
import jax.numpy as jnp
import numpy as np

Points2D = FloatArray['... n 2']
Points3D = FloatArray['... n 3']
MaskedPoints2D = Tuple[Points2D, BoolArray['... n']]
RotationMatrix3D = FloatArray['... 3 3']
RotationMatrix2D = FloatArray['... 2 2']
Angle = FloatArray['...']
Point3D = FloatArray['... 3']
Point2D = FloatArray['... 2']


class Transform3D(dca.DataclassArray):  # pytype: disable=base-class-error
  """SE(3) transformation with 3-DoF translation and 3-DoF rotation."""

  R: RotationMatrix3D  # pylint: disable=invalid-name
  t: Point3D

  @classmethod
  def from_Rt(cls, R: RotationMatrix3D, t: Point3D) -> 'Transform3D':  # pylint: disable=invalid-name
    return cls(R=R, t=t)

  def to_4x4matrix(self) -> FloatArray['... 4 4']:
    mat = jnp.tile(jnp.eye(4), self.shape + (1, 1))  # pytype: disable=attribute-error
    mat = mat.at[..., :3, :3].set(self.R)
    mat = mat.at[..., :3, 3].set(self.t)
    return mat

  @property
  def inv(self) -> 'Transform3D':
    R_inv = self.R.swapaxes(-1, -2)  # pylint: disable=invalid-name
    t_inv = -self.xnp.einsum('...ij,...j->...i', R_inv, self.t)  # pytype: disable=attribute-error
    return self.__class__(R=R_inv, t=t_inv)

  def magnitude(self) -> Tuple[FloatArray, FloatArray]:
    # From https://en.wikipedia.org/wiki/Rotation_matrix#Determining_the_angle
    # trace(R) = 1 + 2 cos(angle)
    trace = self.R.trace(axis1=-2, axis2=-1)
    cos = jnp.clip((trace - 1) / 2, -1, 1)
    dr = jnp.rad2deg(jnp.abs(jnp.arccos(cos)))
    dt = jnp.linalg.norm(self.t, axis=-1)
    return dr, dt

  def transform(self, p3d: Points3D) -> Points3D:
    p3d = self.xnp.einsum('...ij,...nj->...ni', self.R, p3d)  # pytype: disable=attribute-error
    return self.t[..., None, :] + p3d

  def compose(self, other: 'Transform3D') -> 'Transform3D':
    R = self.R @ other.R  # pylint: disable=invalid-name
    t = self.t + self.xnp.einsum('...ij,...j->...i', self.R, other.t)  # pytype: disable=attribute-error
    return self.__class__(R=R, t=t)

  def __matmul__(
      self, other: Union[Points3D, 'Transform3D']
  ) -> Union[Points3D, 'Transform3D']:
    if isinstance(other, (jnp.ndarray, np.ndarray)):
      return self.transform(other)
    elif isinstance(other, Transform3D):
      return self.compose(other)
    else:
      raise TypeError(f'Unexpected type: {type(other)}')


class Transform2D(dca.DataclassArray):  # pytype: disable=base-class-error
  """SE(2) transformation with 2-DoF translation and 1-DoF rotation.

  Attributes:
    angle: Rotation angle, in radians.
    t: 2D translation vector.
  """

  angle: Angle
  t: Point2D

  @classmethod
  def from_radians(cls, angle: Angle, t: Point2D) -> 'Transform2D':
    return cls(angle=angle, t=t)

  @classmethod
  def from_R(cls, R: RotationMatrix2D, t: Point2D) -> 'Transform2D':  # pylint: disable=invalid-name
    xnp = dca.utils.np_utils.get_xnp(R)
    angle = xnp.arctan2(R[..., 1, 0], R[..., 0, 0])
    return cls.from_radians(angle, t)

  @classmethod
  def from_Transform3D(cls, transform: Transform3D) -> 'Transform2D':
    return cls.from_R(transform.R, transform.t[..., :2])

  @property
  def R(self) -> RotationMatrix2D:
    cos = self.xnp.cos(self.angle)  # pytype: disable=attribute-error
    sin = self.xnp.sin(self.angle)  # pytype: disable=attribute-error
    R_flat = self.xnp.stack([cos, -sin, sin, cos], -1)  # pylint: disable=invalid-name  # pytype: disable=attribute-error
    return R_flat.reshape(*self.shape, 2, 2)  # pytype: disable=attribute-error

  def to_3x3matrix(self) -> FloatArray['... 3 3']:
    mat = jnp.tile(jnp.eye(3), self.shape + (1, 1))  # pytype: disable=attribute-error
    mat = mat.at[..., :2, :2].set(self.R)
    mat = mat.at[..., :2, 2].set(self.t)
    return mat

  @property
  def inv(self) -> 'Transform2D':
    R_inv = self.R.swapaxes(-1, -2)  # pylint: disable=invalid-name
    t_inv = -self.xnp.einsum('...ij,...j->...i', R_inv, self.t)  # pytype: disable=attribute-error
    return self.__class__(angle=-self.angle, t=t_inv)

  def magnitude(self) -> Tuple[FloatArray, FloatArray]:
    dr = self.xnp.rad2deg(self.xnp.abs(self.angle)) % 360  # pytype: disable=attribute-error
    dr = self.xnp.minimum(dr, 360 - dr)  # pytype: disable=attribute-error
    dt = self.xnp.linalg.norm(self.t, axis=-1)  # pytype: disable=attribute-error
    return dr, dt

  def transform(self, points: Points2D) -> Points2D:
    points = self.xnp.einsum('...ij,...nj->...ni', self.R, points)  # pytype: disable=attribute-error
    return self.t[..., None, :] + points

  def compose(self, other: 'Transform2D') -> 'Transform2D':
    angle = self.angle + other.angle
    t = self.t + self.xnp.einsum('...ij,...j->...i', self.R, other.t)  # pytype: disable=attribute-error
    return self.__class__(angle=angle, t=t)

  def __matmul__(
      self, other: Union[Points2D, 'Transform2D']
  ) -> Union[Points2D, 'Transform2D']:
    if isinstance(other, (jnp.ndarray, np.ndarray)):
      return self.transform(other)
    elif isinstance(other, Transform2D):
      return self.compose(other)
    else:
      raise TypeError(f'Unexpected type: {type(other)}')


TransformND = Union[Transform3D, Transform2D]


class Camera(dca.DataclassArray):  # pytype: disable=base-class-error
  """Simple pinhole camera.

  The principal point and (un)projected pixel points are expressed in a
  coordinate system whose origin is the top left corner of the top left pixel,
  i.e. half-integer pixel centers (see go/pixelcenters).

  Attributes:
    wh: The size of the image as (width, height).
    f: The focal length, in pixels.
    c: The principal point, in pixels.
    eps: epsilon value for computations that involve zero-clipping.
  """

  wh: FloatArray['... 2']
  f: FloatArray['... 2']
  c: FloatArray['... 2']
  eps = 1e-3

  def scale(self, scale: FloatArray['... 2']) -> 'Camera':
    """Resize the camera by a scaling factor."""
    return self.__class__(
        wh=self.wh * scale, f=self.f * scale, c=self.c * scale
    )

  @jax.jit
  def K(self) -> FloatArray['... 3 3']:
    """Warning: not auto-batched, requires explicit vmap."""
    ret = jnp.eye(3, dtype=self.f.dtype)
    ret = ret.at[[0, 1], [0, 1]].set(self.f)
    ret = ret.at[[0, 1], [2, 2]].set(self.c)
    return ret

  @jax.jit
  def in_image(self, p2d: Points2D) -> BoolArray['... n']:
    """Check if 2D points are within the image boundaries."""
    return jnp.all((p2d >= 0) & (p2d < self.wh[..., None, :]), -1)

  @jax.jit
  def project(self, p3d: Points3D) -> MaskedPoints2D:
    """Project 3D points into the camera plane and check for visibility."""
    z = p3d[..., -1]
    valid = z >= self.eps
    z = z.clip(min=self.eps)[..., None]
    p2d = p3d[..., :-1] / z
    return p2d, valid

  @jax.jit
  def denormalize(self, p2d: Points2D) -> Points2D:
    """Convert normalized 2D coordinates into pixel coordinates."""
    return p2d * self.f[..., None, :] + self.c[..., None, :]

  @jax.jit
  def normalize(self, p2d: Points2D) -> Points2D:
    return (p2d - self.c[..., None, :]) / self.f[..., None, :]

  @jax.jit
  def world2image(self, p3d: Points3D) -> MaskedPoints2D:
    p2d, visible = self.project(p3d)
    p2d = self.denormalize(p2d)
    valid = visible & self.in_image(p2d)
    return p2d, valid


class FisheyeCamera(Camera):
  """Camera with fisheye distortion.

  Attributes:
    k_radial: Radial distortion coefficients.
    max_fov: Maximum field-of-view, in radians.
  """

  k_radial: FloatArray['... 3']
  max_fov: FloatArray['...']

  @classmethod
  def from_dict(cls, intrinsics: Dict[str, Any]) -> 'FisheyeCamera':
    K = intrinsics['K']  # pylint: disable=invalid-name
    xnp = dca.utils.np_utils.get_xnp(K)
    wh = (intrinsics['image_width'], intrinsics['image_height'])
    wh = xnp.stack(wh, -1).astype(K.dtype)
    f = K[..., [0, 1], [0, 1]]
    c = K[..., [0, 1], [2, 2]]
    k_radial = intrinsics['distortion']['radial']
    max_fov = intrinsics.get('maxfov')
    if max_fov is None:
      # Backward compatibility: the max FoV is generally 115deg on H1 data.
      max_fov = xnp.full(wh.shape[:-1], np.deg2rad(115.0), K.dtype)
    return cls(wh=wh, f=f, c=c, k_radial=k_radial, max_fov=max_fov)

  def scale(self, scale: FloatArray['... 2']) -> 'Camera':
    """Resize the camera by a scaling factor."""
    return self.__class__(
        wh=self.wh * scale,
        f=self.f * scale,
        c=self.c * scale,
        k_radial=self.k_radial,
        max_fov=self.max_fov,
    )

  @jax.jit
  def distort_points(self, p2d: Points2D) -> MaskedPoints2D:
    radius2 = jnp.sum(p2d**2, axis=-1)
    in_center = radius2 < self.eps**2
    radius = jnp.sqrt(jnp.where(in_center, self.eps**2, radius2))
    theta = jnp.arctan(radius)
    theta2 = theta**2
    offset = sum(self.k_radial[..., i] * theta2 ** (i + 1) for i in range(3))
    dist = (offset + 1) * theta / radius
    dist = jnp.where(in_center, 1.0, dist)
    p2d_dist = p2d * dist[..., None]
    valid = in_center | ((radius < jnp.tan(0.5 * self.max_fov)) & (dist > 0))
    return p2d_dist, valid

  @jax.jit
  def world2image(self, p3d: Points3D) -> MaskedPoints2D:
    p2d, visible = self.project(p3d)
    p2d, valid = self.distort_points(p2d)
    p2d = self.denormalize(p2d)
    valid = visible & valid & self.in_image(p2d)
    return p2d, valid
