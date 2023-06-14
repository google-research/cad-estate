# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: spopov@google.com (Stefan Popov)
"""Functions to compute transformation matrices."""

import torch as t
from torch.nn import functional as F


def scale(v: t.Tensor) -> t.Tensor:
  """Computes homogeneous scale matrices from scale vectors.

  Args:
    v: Scale vectors, `float32[B*, N]`

  Returns:
    Scale matrices, `float32[B*, N+1, N+1]`
  """
  v = t.as_tensor(v, dtype=t.float32)
  batch_dims = v.shape[:-1]
  v = v.reshape([-1, (v.shape[-1])])

  index_batch_flat = t.arange(v.shape[0], dtype=t.int64, device=v.device)
  index_diag = t.arange(v.shape[1], dtype=t.int64, device=v.device)
  index_batch, index_diag = t.meshgrid(index_batch_flat, index_diag,
                                       indexing="ij")
  index_batch = index_batch.reshape([-1])
  index_diag = index_diag.reshape([-1])

  result = v.new_zeros([v.shape[0], v.shape[1] + 1, v.shape[1] + 1])
  result[index_batch, index_diag, index_diag] = v.reshape([-1])
  result[index_batch_flat, v.shape[-1], v.shape[-1]] = 1
  result = result.reshape(batch_dims + result.shape[-2:])
  return result


def translate(v: t.Tensor) -> t.Tensor:
  """Computes a homogeneous translation matrices from translation vectors.

  Args:
    v: Translation vectors, `float32[B*, N]`

  Returns:
    Translation matrices, `float32[B*, N+1, N+1]`
  """
  result = t.as_tensor(v, dtype=t.float32)
  dimensions = result.shape[-1]
  result = result[..., None, :].transpose(-1, -2)
  result = t.constant_pad_nd(result, [dimensions, 0, 0, 1])
  id_matrix = t.diag(result.new_ones([dimensions + 1]))
  id_matrix = id_matrix.expand_as(result)
  result = result + id_matrix
  return result


def rotate(
    angle: t.Tensor,
    axis: t.Tensor,
) -> t.Tensor:
  """Computes a 3D rotation matrices from angle and axis inputs.

  The formula used in this function is explained here:
  https://en.wikipedia.org/wiki/Rotation_matrix#Conversion_from_and_to_axis–angle

  Args:
    angle: The rotation angles, `float32[B*]`
    axis: The rotation axes, `float32[B*, 3]`

  Returns:
    The rotation matrices, `float32[B*, 4, 4]`
  """
  axis = t.as_tensor(axis, dtype=t.float32)
  angle = t.as_tensor(angle, dtype=t.float32)
  axis = F.normalize(axis, dim=-1)
  sin_axis = t.sin(angle)[..., None] * axis
  cos_angle = t.cos(angle)
  cos1_axis = (1.0 - cos_angle)[..., None] * axis
  _, axis_y, axis_z = t.unbind(axis, dim=-1)
  cos1_axis_x, cos1_axis_y, _ = t.unbind(cos1_axis, dim=-1)
  sin_axis_x, sin_axis_y, sin_axis_z = t.unbind(sin_axis, dim=-1)
  tmp = cos1_axis_x * axis_y
  m01 = tmp - sin_axis_z
  m10 = tmp + sin_axis_z
  tmp = cos1_axis_x * axis_z
  m02 = tmp + sin_axis_y
  m20 = tmp - sin_axis_y
  tmp = cos1_axis_y * axis_z
  m12 = tmp - sin_axis_x
  m21 = tmp + sin_axis_x
  zero = t.zeros_like(m01)
  one = t.ones_like(m01)
  diag = cos1_axis * axis + cos_angle[..., None]
  diag_x, diag_y, diag_z = t.unbind(diag, dim=-1)

  matrix = t.stack((diag_x, m01, m02, zero, m10, diag_y, m12, zero, m20, m21,
                    diag_z, zero, zero, zero, zero, one), dim=-1)
  output_shape = axis.shape[:-1] + (4, 4)
  result = matrix.reshape(output_shape)
  return result


def transform_points_homogeneous(points: t.Tensor, matrix: t.Tensor,
                                 w: float) -> t.Tensor:
  """Transforms 3D points with a homogeneous matrix.

  Args:
    points: The points to transform, `float32[B*, N, 3]`
    matrix: The transformation matrices, `float32[B*, 4, 4]`
    w: The W value to add to the points to make them homogeneous. Should be 1
      for affine points and 0 for vectors.

  Returns:
    The transformed points in homogeneous space (with a 4th coordinate),
    `float32[B*, N, 4]`
  """
  batch_dims = points.shape[:-2]
  # Fold all batch dimensions into a single one
  points = points.reshape([-1] + list(points.shape[-2:]))
  matrix = matrix.reshape([-1] + list(matrix.shape[-2:]))

  points = t.constant_pad_nd(points, [0, 1], value=w)
  result = t.einsum("bnm,bvm->bvn", matrix, points)
  result = result.reshape(batch_dims + result.shape[-2:])

  return result


def transform_mesh(mesh: t.Tensor, matrix: t.Tensor,
                   vertices_are_points=True) -> t.Tensor:
  """Transforms a single 3D mesh.

  Args:
    mesh: The mesh's triangle vertices, `float32[B*, N, 3, 3]`
    matrix: The transformation matrix, `float32[B*, 4, 4]`
    vertices_are_points: Whether to interpret the vertices as affine points
      or vectors.

  Returns:
    The transformed mesh, `float32[B*, N, 3, 3]`
  """
  original_shape = mesh.shape
  mesh = mesh.reshape([-1, mesh.shape[-3] * 3, 3])
  matrix = matrix.reshape([-1, 4, 4])
  w = 1 if vertices_are_points else 0
  mesh = transform_points_homogeneous(mesh, matrix, w=w)
  if vertices_are_points:
    mesh = mesh[..., :3] / mesh[..., 3:4]
  else:
    mesh = mesh[..., :3]

  return mesh.reshape(original_shape)


def transform_points(points: t.Tensor, matrix: t.Tensor) -> t.Tensor:
  """Transforms points.
  Args:
    points: The points to transform, `float32[B*, N, 3]`
    matrix: Transformation matrices, `float32[B*, 4, 4]`
  Result:
    The transformed points, `float32[B*, N, 3]`
  """
  result = transform_points_homogeneous(points, matrix, w=1)
  result = result[..., :3] / result[..., 3:4]
  return result


def chain(transforms: list[t.Tensor], reverse=True) -> t.Tensor:
  """Chains transformations expressed as matrices.

  Args:
    transforms: The list of transformations to chain
    reverse: The order in which transformations are applied. If true, the last
      transformation is applied first (which matches matrix multiplication
      order). False matches natural order, where the first transformation is
      applied first.

  Returns:
    Matrix combining all transformations.

  """
  assert transforms
  if not reverse:
    transforms = transforms[::-1]
  result = transforms[0]
  for transform in transforms[1:]:
    result = result @ transform
  return result


def gl_projection_matrix_from_intrinsics(  #
    width: t.Tensor, height: t.Tensor, fx: t.Tensor, fy: t.Tensor, cx: t.Tensor,
    cy: t.Tensor, znear: float = 0.001, zfar: float = 20.) -> t.Tensor:
  """Computes the camera projection matrix for rendering square images.

  Args:
    width: Image's `width`, `float32[B*]`.
    height: Image's `heigh`t,` float32[B*]`.
    fx: Camera's `fx`, `float32[B*]`.
    fy: Camera's `fy`, `float32[B*]`.
    cx: Camera's `cx`, `float32[B*]`.
    cy: Camera's `cy`, `float32[B*]`.
    znear: The near plane location.
    zfar: The far plane location.

  Returns:
    World to OpenGL's normalized device coordinates transformation matrices,
    `float32[B*, 4, 4]`.
  """

  z = t.zeros_like(t.as_tensor(fx))
  o = t.ones_like(z)
  zn = znear * o
  zf = zfar * o
  # yapf: disable
  result = [
      2 * fx / width, z, 2 * (cx / width) - 1, z,
      z, 2 * fy / height, 2 * (cy / height) - 1, z,
      z, z, (zf + zn) / (zf - zn), -2 * zn * zf / (zf - zn),
      z, z, o, z
  ]
  # yapf: enable
  result = t.stack([t.as_tensor(v, dtype=t.float32)
                    for v in result]).reshape((4, 4) + z.shape)

  result = result.permute(tuple(range(len(result.shape)))[2:] + (0, 1))
  return result


def quaternion_to_rotation_matrix(q: t.Tensor) -> t.Tensor:
  """Computes a rotation matrix from a quaternion.

  Args:
    q: Rotation quaternions, float32[B*, 4]

  Returns:
    Rotation matrices, float32[B, 4, 4]

  """
  q = t.as_tensor(q, dtype=t.float32)
  w, x, y, z = t.unbind(q, dim=-1)
  zz = t.zeros_like(z)
  oo = t.ones_like(z)
  s = 2.0 / (q * q).sum(dim=-1)
  # yapf: disable
  return t.stack([
      1 - s * (y ** 2 + z ** 2), s * (x * y - z * w), s * (x * z + y * w), zz,
      s * (x * y + z * w), 1 - s * (x ** 2 + z ** 2), s * (y * z - x * w), zz,
      s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x ** 2 + y ** 2), zz,
      zz, zz, zz, oo
  ], dim=-1).reshape(q.shape[:-1] + (4, 4))
  # yapf: enable
