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
#
"""High level routines for rendering scenes."""

import io
from importlib import resources
from typing import Iterable

import numpy as np
import PIL.Image
import torch as t

from cad_estate import misc_util
from cad_estate.gl import rasterizer
from cad_estate.gl import shaders

InputTensor = t.Tensor | np.ndarray | int | float | Iterable | None


def load_textures(
    encoded_images: Iterable[bytes],
    texture_size: tuple[int, int],
) -> tuple[t.Tensor, t.Tensor]:
  """Composes a texture array from encoded images contained in strings.

  Args:
    encoded_images: The encoded images, string[num_images]. Each entry must
      either be a valid image (e.g. PNG or JPEG) or an empty string.
    texture_size: Tuple (height, width) giving the desired dimensions of the
      output texture array.

  Returns:
    texture_array: uint8[num_non_empty_images, height, width, 3] tensor
      containing the decoded images from the non-empty entries in
      encoded_images. All images are scaled to the desired height and width
      and flipped along the Y axis.
    image_indices: int32[num_images] tensor that defines the mapping between
      encoded_images and texture_array. The j-th entry in encoded_images
      will be decoded to texture_array[image_indices[j]]. If encoded_images[j]
      is empty then image_indices[j] = -1.
  """
  # The empty string maps to -1
  image_to_index = {b"": -1}
  image_indices = []
  height, width = texture_size
  texture_array = []
  for encoded_image in encoded_images:
    if encoded_image not in image_to_index:
      image_to_index[encoded_image] = len(image_to_index) - 1
      pil_image = (PIL.Image.open(io.BytesIO(encoded_image))
                  )  # type: PIL.Image.Image
      image = np.array(
          pil_image.convert("RGB").resize((width, height),
                                          resample=PIL.Image.BICUBIC))
      assert (len(image.shape) == 3 and image.shape[-1] == 3
              and image.dtype == np.uint8)
      texture_array.append(image)
    image_indices.append(image_to_index[encoded_image])

  image_indices = misc_util.to_tensor(image_indices, t.int32, "cpu")
  if texture_array:
    texture_array = misc_util.to_tensor(texture_array, t.uint8, "cpu")
  else:
    texture_array = t.zeros([1, 1, 1, 3], dtype=t.uint8)
  texture_array = texture_array.flip(1).contiguous()

  return texture_array, image_indices


def render_scene(
    vertex_positions: InputTensor,
    view_projection_matrix: InputTensor,
    image_size: tuple[int, int] = (256, 256),
    *,
    normals: InputTensor = None,
    vertex_colors: InputTensor = None,
    tex_coords: InputTensor = None,
    material_ids: InputTensor = None,
    diffuse_coefficients: InputTensor = None,
    diffuse_textures: InputTensor = None,
    diffuse_texture_indices: InputTensor = None,
    specular_coefficient: InputTensor = None,
    ambient_coefficients: InputTensor = None,
    cull_back_facing=True,
    light_position: InputTensor = None,
    light_color: InputTensor = (1.0, 1.0, 1.0),
    ambient_light_color: InputTensor = (0.2, 0.2, 0.2),
    clear_color: InputTensor = (0, 0, 0, 1),
    output_type=t.uint8,
    vertex_shader=None,
    geometry_shader=None,
    fragment_shader=None,
    debug_io_buffer=None,
    return_rgb=True,
    device=None,
):
  """Renders the given scene.

  Args:
    vertex_positions: The triangle geometry, specified through the triangle
      vertex positions, float32[num_triangles, 3, 3]
    view_projection_matrix: The view projection matrix, float32[4, 4]
    image_size: Desired output image size, (height, width),
    normals: Per-vertex shading normals, float32[num_triangles, 3, 3]. If set to
      None, normals will be computed from the vertex positions.
    vertex_colors: Optional per-vertex colors, float32[num_triangles, 3, 3].
    tex_coords: Texture coordinate, float32[num_triangles, 3, 2]. If set to
      None, all texture coordinates will be 0.
    material_ids: Per-triangle material indices used to index in the various
      coefficient tensors below, int32[num_triangles]. If set to None, all
      triangles will have the same default material.
    diffuse_coefficients: The diffuse coefficients, one per material,
      float32[num_materials, 3]. Cannot be None if material_ids is not None.
      Must be None if material_ids is None.
    diffuse_textures: uint8[num_textures, height, width, 3]. Can be None if
      there are no textures used in the mesh.
    diffuse_texture_indices: Diffuse texture indices, one per material,
      int32[num_materials]. If set to None, the texture indices for all
      materials will be -1.
    specular_coefficient: Specular coefficients, one per material,
      float32[num_materials, 4]. The first 3 channels are the R, G, and B
      specular coefficients, the last channel is the specular power. If set to
      None, R, G, and B will be 0 for all materials and power will be 2048.
    ambient_coefficients: float32[num_materials, 3]. The ambient coefficients.
      If None, all ambient coefficient will be 0.05.
    cull_back_facing: whether to cull backfacing triangles.
    light_position: float32[3], the light position. If set to None, the light
      will be placed at the camera origin.
    light_color: The light diffuse RGB color, float32[3]
    ambient_light_color: The light ambient RGB color, float32[3]
    clear_color: The RGB color to use when clearing the image, float32[3]
    output_type: The desired output type. Either tf.uint8 or tf.float32.
    vertex_shader: The vertex shader to use. If empty, uses a default shader.
    geometry_shader: The geometry shader. If empty, uses a default shader.
    fragment_shader: The fragment shader. If empty, uses a default shader.
    debug_io_buffer: Aids debugging of shaders. Shaders can communicate with
      host programs through OpenGL input/output buffers. Any tensor passed in
      this argument will be forwarded to the shaders as buffer with name
      "debug_io".
    return_rgb: If true, returns a 3 channel image, otherwise returns a 4
      channel image.
    device: The index of the GPU to use, given as CUDA device

  Returns:
    The rendered image, dt[height, width, c] where dt is either float32 or uint8
    depending on the value of output_type and c is either 3 or 4, depending on
    return_rgb. If the debug_io_buffer argument was not None, returns a
    tuple containing the rendered image, and the shader output from the
    "debug_io" buffer. The second element of the tuple has the same shape
    and type as debug_io_buffer.

  """
  if device is None:
    device = t.cuda.current_device()

  height, width = image_size
  vertex_positions = misc_util.to_tensor(vertex_positions, t.float32, device)
  assert (len(vertex_positions.shape) == 3
          and vertex_positions.shape[1:] == (3, 3))
  num_triangles = vertex_positions.shape[0]

  view_projection_matrix = misc_util.to_tensor(view_projection_matrix,
                                               t.float32, device)
  assert view_projection_matrix.shape == (4, 4)

  has_normals = True
  if normals is None:
    # normals = t.zeros_like(vertex_positions)
    normals = t.zeros([1, 3, 3], device=device)
    has_normals = False
  else:
    assert normals.shape == (num_triangles, 3, 3)

  if vertex_colors is None:
    vertex_colors = t.zeros((1, 3, 3), dtype=t.float32, device=device)
    has_vertex_colors = False
  else:
    has_vertex_colors = True
    assert vertex_colors.shape == (num_triangles, 3, 3)

  if tex_coords is None:
    tex_coords = t.zeros([1, 3, 2], dtype=t.float32)
  else:
    tex_coords = misc_util.to_tensor(tex_coords, t.float32, device)
    assert tex_coords.shape == (num_triangles, 3, 2)

  if material_ids is None:
    material_ids = t.zeros([num_triangles], dtype=t.int32)
  material_ids = misc_util.to_tensor(material_ids, t.int32, device)
  assert material_ids.shape == (num_triangles,)
  num_used_materials = material_ids.max().cpu().numpy() + 1  # type: int

  def create_coefficient_array(cur_tensor: InputTensor, num_channels,
                               default_value):
    arr = cur_tensor
    if arr is None:
      arr = (
          t.ones([num_used_materials, num_channels], dtype=t.float32) *
          t.tensor(default_value))
    arr = misc_util.to_tensor(arr, t.float32, device)
    assert len(arr.shape) == 2
    arr = arr[:num_used_materials]
    assert arr.shape == (num_used_materials, num_channels)
    return arr

  diffuse_coefficients = create_coefficient_array(diffuse_coefficients, 3, 0.8)
  ambient_coefficients = create_coefficient_array(ambient_coefficients, 3, 0.05)
  specular_coefficient = create_coefficient_array(specular_coefficient, 4,
                                                  (0, 0, 0, 2048.0))
  if diffuse_texture_indices is None:
    diffuse_texture_indices = t.ones([num_used_materials], dtype=t.int32) * -1
  diffuse_texture_indices = misc_util.to_tensor(diffuse_texture_indices,
                                                t.int32, device)
  assert len(diffuse_texture_indices.shape) == 1
  diffuse_texture_indices = diffuse_texture_indices[:num_used_materials]
  assert diffuse_texture_indices.shape == (num_used_materials,)
  num_used_textures = diffuse_texture_indices.max().cpu().numpy() + 1
  num_used_textures = max(num_used_textures, 1)

  if diffuse_textures is None:
    diffuse_textures = t.ones([num_used_textures, 1, 1, 3], dtype=t.uint8)
  diffuse_textures = misc_util.to_tensor(diffuse_textures, t.uint8, device)
  assert len(diffuse_textures.shape) == 4
  diffuse_textures = diffuse_textures[:num_used_textures]
  assert (diffuse_textures.shape[0] == num_used_textures
          and diffuse_textures.shape[3] == 3)

  # The projection center transforms to (0, 0, -a, 0) in NDC space, assuming
  # default GL conventions for the projection matrix (i.e. its last column in
  # (0, 0, -a, 0). To recover its position in world space, we multiply by
  # the inverse view-projection matrix. Tha value of `a` doesn't matter, we
  # use 1.
  camera_position = t.mv(
      t.inverse(view_projection_matrix),
      t.tensor([0, 0, -1, 0], dtype=t.float32, device=device))
  camera_position = camera_position[:3] / camera_position[3]

  if light_position is None:
    light_position = camera_position
  light_position = misc_util.to_tensor(light_position, t.float32, device)
  assert light_position.shape == (3,)

  light_color = misc_util.to_tensor(light_color, t.float32, device)
  assert light_color.shape == (3,)

  ambient_light_color = misc_util.to_tensor(ambient_light_color, t.float32,
                                            device)
  assert ambient_light_color.shape == (3,)

  ambient_coefficients = t.constant_pad_nd(ambient_coefficients, [0, 1])
  diffuse_coefficients = t.cat([
      diffuse_coefficients,
      diffuse_texture_indices.to(t.float32)[:, np.newaxis]
  ], -1)
  materials = t.cat(
      [ambient_coefficients, diffuse_coefficients, specular_coefficient],
      dim=-1)

  render_args = [
      rasterizer.Uniform("view_projection_matrix", view_projection_matrix),
      rasterizer.Uniform("light_position", light_position),
      rasterizer.Uniform("has_normals", has_normals),
      rasterizer.Uniform("has_vertex_colors", has_vertex_colors),
      rasterizer.Uniform("has_texcoords", True),
      rasterizer.Buffer(0, vertex_positions.reshape([-1])),
      rasterizer.Buffer(1, normals.reshape([-1])),
      rasterizer.Buffer(2, vertex_colors.reshape([-1])),
      rasterizer.Buffer(3, tex_coords.reshape([-1])),
      rasterizer.Buffer(4, material_ids.reshape([-1])),
      rasterizer.Buffer(5, materials.reshape([-1])),
      rasterizer.Texture("textures", diffuse_textures, bind_as_array=True),
      rasterizer.Uniform("light_color", light_color),
      rasterizer.Uniform("camera_position", camera_position),
      rasterizer.Uniform("ambient_light_color", ambient_light_color),
      rasterizer.Uniform("cull_backfacing", cull_back_facing),
  ]

  if debug_io_buffer is not None:
    render_args.append(rasterizer.Buffer(5, debug_io_buffer, is_io=True))

  if not geometry_shader:
    geometry_shader = resources.read_text(shaders, "triangle_renderer.geom")
  if not vertex_shader:
    vertex_shader = resources.read_text(shaders, "noop.vert")
  if not fragment_shader:
    fragment_shader = resources.read_text(shaders,
                                          "point_light_illumination.frag")

  result = rasterizer.gl_simple_render(
      rasterizer.RenderInput(
          num_points=num_triangles,
          arguments=render_args,
          output_resolution=(height, width),
          clear_color=clear_color,
          output_type=output_type,
          vertex_shader=vertex_shader,
          geometry_shader=geometry_shader,
          fragment_shader=fragment_shader,
      ), cuda_device=device)

  c = 3 if return_rgb else 4
  if debug_io_buffer is None:
    return result[..., :c]
  else:
    return result[..., :c], render_args[-1].value
