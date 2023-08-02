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
"""Library for loading aligned objects."""

import dataclasses
import io
import json

import numpy as np
import torch as t

from cad_estate import file_system as fs
from cad_estate import frames as frame_lib
from cad_estate import input_file_structures as input_struct_lib
from cad_estate import misc_util
from cad_estate import transformations as transform_lib

# We use the following symbols to describe tensors below:
# NUM_OBJ     : Number of objects
# NUM_OBJ_TRI : Total number of triangles in all objects
# NUM_FRAMES  : Number of loaded frames in a video


@dataclasses.dataclass
class Objects(misc_util.TensorContainerMixin):
  """3D objects inside a video clip."""

  clip_name: str
  """The RealEstate-10K clip name.
  Uniquely identifies the scene. Consists of a YouTube video ID,
  followed by "_", and then by a start timestamp
  """

  triangles: t.Tensor
  """The triangles of all objects, `float32[NUM_OBJ_TRI, 3, 3]`.
  Coordinates are in world space. The second dimension is triangle vertices,
  the third dimension is 3D coordinates (`X`, `Y`, `Z`). Triangles of each
  object instance are stored sequentially.
  """

  num_tri: t.Tensor
  """The number of triangles in each object, `int64[NUM_OBJ]`.
  The first `num_tri[0]` triangles in `triangles` belong to the first
  object, the next `num_tri[1]` belong to the second object, and so on.
  """

  object_to_world: t.Tensor
  """Object->world transformation matrices, `float32[NUM_OBJ, 4, 4]`.
  Applying the inverse matrix to the triangles of an object will result in the
  original ShapeNet geometry.
  """

  mesh_ids: np.ndarray
  """The ShapeNet model IDs for the objects, `str[NUM_OBJ]`."""

  symmetries: t.Tensor
  """The object symmetries, `int64[NUM_OBJ]`.
  A value of N means that the object is N-way symmetric. That is, rotating it
  by `pi * 2 / N` degrees, results in the same geometry. This value is
  1 for asymmetric objects.
  """

  labels: np.ndarray
  """The semantic labels of the objects, `int64[NUM_OBJ]`."""

  from_automatic_track: t.Tensor
  """Whether an object was created from an automatic or a manual track,
  `bool[NUM_OBJ]`."""


@dataclasses.dataclass
class ShapeNetMetadata:
  """Metadata for the ShapeNet dataset."""

  shapenet_root_dir: str
  """Root directory for all ShapeNet shapes."""

  label_synset: list[str]
  """Maps integer labels to ShapeNet Synsets."""

  label_human_readable: list[str]
  """Maps integer labels to human readable class names."""

  synset_to_index: dict[str, int]
  """Maps ShapeNet Synsets to integer labels"""

  symmetry_dict: dict[str, int]
  """The symmetry for each ShapeNet shape. Symmetry `K` means that the shape
  does not change when rotated by `360/K` degrees around the up axis.
  """


def load_shapenet_metadata(shapenet_root_dir: str) -> ShapeNetMetadata:
  """Loads saved ShapeNet metadata."""
  labels = json.loads(
      fs.read_text(fs.join(shapenet_root_dir, "shapenet_classes.json")))
  label_synset = [v["id"] for v in labels]
  label_human_readable = [v["human_readable"] for v in labels]
  synset_to_index = {k: i for i, k in enumerate(label_synset)}
  symmetry_dict = json.loads(
      fs.read_text(fs.join(shapenet_root_dir, "symmetry_dict.json")))

  return ShapeNetMetadata(shapenet_root_dir=shapenet_root_dir,
                          label_synset=label_synset,
                          label_human_readable=label_human_readable,
                          synset_to_index=synset_to_index,
                          symmetry_dict=symmetry_dict)


async def load_objects(obj_json: input_struct_lib.ObjectsFile,
                       shapenet_meta: ShapeNetMetadata):
  """Loads objects from a JSON."""

  clip_name = obj_json["clip_name"]

  obj_paths, obj_labels, obj_mesh_ids = [], [], []
  from_automatic_track = []
  for obj in obj_json["objects"]:
    if "translation" in obj:
      obj_paths.append(
          fs.join(shapenet_meta.shapenet_root_dir, obj["class"],
                  obj["cad_id"] + ".npz"))
      obj_mesh_ids.append(obj["cad_id"])
    else:
      obj_paths.append("")
      obj_mesh_ids.append("")
    obj_labels.append(shapenet_meta.synset_to_index[obj["class"]])
    from_automatic_track.append(obj["is_track_automatic"])

  obj_labels = t.as_tensor(obj_labels, dtype=t.int64)
  obj_mesh_ids = np.array(obj_mesh_ids, dtype=np.str_)
  from_automatic_track = t.as_tensor(from_automatic_track, dtype=t.bool)

  unique_cad_paths = sorted(set([v for v in obj_paths if v]))
  npz_bytes = await fs.read_all_bytes_async(unique_cad_paths)
  meshes = [np.load(io.BytesIO(v))["vertices"] for v in npz_bytes]
  meshes = [t.as_tensor(v, dtype=t.float32) for v in meshes]
  meshes = {k: v for k, v in zip(unique_cad_paths, meshes)}

  obj_num_tri, obj_triangles, object_to_world = [], [], []
  obj_symmetries = []
  for obj_idx, obj in enumerate(obj_json["objects"]):
    if "translation" in obj:
      mesh = meshes[obj_paths[obj_idx]]
      obj_num_tri.append(mesh.shape[0])
      o2w = [
          transform_lib.translate(obj["translation"]),
          transform_lib.quaternion_to_rotation_matrix(obj["rotation"]),
          transform_lib.scale(obj["scale"]),
      ]
      if obj["is_mirrored"]:
        o2w.append(transform_lib.scale([-1, 1, 1]))
      o2w = transform_lib.chain(o2w)
      object_to_world.append(o2w)
      obj_triangles.append(transform_lib.transform_mesh(mesh, o2w))
      obj_symmetries.append(
          shapenet_meta.symmetry_dict[f"{obj['class']}_{obj['cad_id']}"])
    else:
      obj_num_tri.append(0)
      object_to_world.append(t.eye(4))
      obj_symmetries.append(1)

  obj_num_tri = t.as_tensor(obj_num_tri, dtype=t.int64)
  if obj_triangles:
    obj_triangles = t.concat(obj_triangles, 0)
    object_to_world = t.stack(object_to_world)
    obj_symmetries = t.as_tensor(obj_symmetries, dtype=t.int64)
  else:
    obj_triangles = t.empty([0, 3, 3], dtype=t.float32)
    object_to_world = t.empty([0, 4, 4], dtype=t.float32)
    obj_symmetries = t.empty([0], dtype=t.int64)

  return Objects(clip_name=clip_name, triangles=obj_triangles,
                 num_tri=obj_num_tri, object_to_world=object_to_world,
                 labels=obj_labels, mesh_ids=obj_mesh_ids,
                 symmetries=obj_symmetries,
                 from_automatic_track=from_automatic_track)


def load_track_boxes(obj_json: input_struct_lib.ObjectsFile,
                     frames: frame_lib.Frames):
  """Loads the annotated object tracks, `float32[NUM_OBJ, NUM_FRAMES, 4]`.
  `result[i, j]` contains the bounding box (`ymin, xmin, ymax, xmax`) of
  object `i` on frame `j`. Coordinates are relative, in the range `[0, 1]`.
  If there is no track annotation for an object/frame pair, all box
  coordinates are set to -1.
  """
  num_obj = len(obj_json["objects"])
  num_frames = frames.frame_timestamps.shape[0]
  track_boxes = t.full((num_obj, num_frames, 4), -1, dtype=t.float32)
  time_stamp_to_index = {
      int(v): i for i, v in enumerate(frames.frame_timestamps)
  }
  for obj_idx, obj in enumerate(obj_json["objects"]):
    for track_entry in obj["track"]:
      frame_idx = time_stamp_to_index.get(track_entry["timestamp"], -1)
      if frame_idx >= 0:
        track_boxes[obj_idx, frame_idx] = t.as_tensor(track_entry["box"],
                                                      dtype=t.float32)
  return track_boxes
