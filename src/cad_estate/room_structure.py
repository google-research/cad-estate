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
"""Library for loading room structures."""

import asyncio
import dataclasses
import re

import torch as t

from cad_estate import file_system as fs
from cad_estate import frames as frame_lib
from cad_estate import input_file_structures as input_struct_lib
from cad_estate import misc_util

# We use the following symbols to describe tensors below:
# NUM_FRAMES      : Number of loaded frames in a video
# NUM_STRUCT_ELEM : Number of structural elements
# NUM_STRUCT_TRI  : The total number of triangles in the room structure
# NUM_ANN_FRAMES  : The number of annotated frames
# AH, AW          : Height and width of the annotations

STRUCTURAL_ELEMENT_TYPES = ["<ignore>", "wall", "floor", "ceiling", "slanted"]


@dataclasses.dataclass
class RoomStructure(misc_util.TensorContainerMixin):
  """3D structural elements for a clip."""

  clip_name: str
  """The RealEstate-10K clip name.
  Uniquely identifies the scene. Consists of a YouTube video ID,
  followed by "_", and then by a start timestamp
  """

  triangles: t.Tensor
  """Structural element triangles, `float32[NUM_STRUCT_TRI, 3, 3]`.
  Triangles of each structural element are stored sequentially.
  """

  num_tri: t.Tensor
  """Number of triangles for each structural element, `int64[NUM_STRUCT_ELEM]`.
  The first `num_tri[0]` triangles in `triangles` belong to the first
  structural element, the next `num_tri[1]` -- to the second, and so on.
  """

  triangle_flags: t.Tensor
  """Additional per-triangle flags, `int64[NUM_STRUCT_TRI, 3]`.
  Bit 1 indicates the triangle is part of a window frame. Bit 2 -- part of a
  closed door.
  """

  labels: t.Tensor
  """Semantic labels of the structural elements, `int64[NUM_STRUCT_ELEM]`.
  STRUCTURAL_ELEMENT_TYPES maps these to class names.
  """

  annotated_timestamps: t.Tensor
  """Timestamps of the annotated frames, `int64[NUM_ANN_FRAMES]`."""


@dataclasses.dataclass
class StructureAnnotations(misc_util.TensorContainerMixin):
  """Structural element annotations."""

  structural_element_masks: t.Tensor
  """Amodal structural element masks, `uint8[NUM_FRAMES, AH, AW]`."""

  visible_parts_masks: t.Tensor
  """Structural elements visible parts, `uint8[NUM_FRAMES, AH, AW]`."""


def load_room_structure(struct_npz: input_struct_lib.RoomStructureFile):
  """Loads room structures."""

  # Load the frame information
  clip_name = struct_npz["clip_name"].item()

  return RoomStructure(
      clip_name=clip_name,
      triangles=t.as_tensor(struct_npz["layout_triangles"], dtype=t.float32),
      triangle_flags=t.as_tensor(struct_npz["layout_triangle_flags"],
                                 dtype=t.int64),
      num_tri=t.as_tensor(struct_npz["layout_num_tri"], dtype=t.int64),
      labels=t.as_tensor(struct_npz["layout_labels"], dtype=t.int64),
      annotated_timestamps=t.as_tensor(struct_npz["annotated_timestamps"],
                                       dtype=t.int64),
  )


async def _load_annotation(target_ref: list[t.Tensor], num_frames: int,
                           index: int, path: str):
  img = await frame_lib.read_and_decode_image(path)
  if img.dtype == t.int16:
    if img.min() < 0 or img.max() > 255:
      raise ValueError("Mask contains values outside of [0, 255].")
    img = img.to(t.uint8)
  target, = target_ref
  if target is None:
    h, w = img.shape
    target_ref[0] = t.full([num_frames, h, w], -1, dtype=t.uint8)
    target, = target_ref
  target[index] = img


async def load_annotations(room: RoomStructure, frames: frame_lib.Frames,
                           annotation_dir: str, raw: bool = False):
  """Loads the room structure annotations."""

  assert frames.clip_name == room.clip_name
  num_frames, c, h, w = frames.frame_images.shape
  assert c == 3

  structure_ref, visible_ref = [None], [None]

  annotated_timestamps = set(room.annotated_timestamps.tolist())
  video_name = re.match(r"^(.+)_\d+$", room.clip_name).group(1)
  prefix = "raw" if raw else "processed"

  tasks = []
  for i, timestamp in enumerate(frames.frame_timestamps.tolist()):
    if timestamp not in annotated_timestamps:
      continue
    root_dir = fs.join(annotation_dir, room.clip_name, "structure_annotations")
    struct_path = fs.join(root_dir,
                          f"{prefix}_structure_{video_name}_{timestamp}.png")
    tasks.append(_load_annotation(structure_ref, num_frames, i, struct_path))
    visible_path = fs.join(root_dir,
                           f"{prefix}_visible_{video_name}_{timestamp}.png")
    tasks.append(_load_annotation(visible_ref, num_frames, i, visible_path))
  if tasks:
    await asyncio.gather(*tasks)
    structure_masks, = structure_ref
    visible_masks, = visible_ref
  else:
    structure_masks = t.full([num_frames, h, w], -1, dtype=t.uint8)
    visible_masks = t.full([num_frames, h, w], -1, dtype=t.uint8)

  return StructureAnnotations(structural_element_masks=structure_masks,
                              visible_parts_masks=visible_masks)


def annotated_frames_mask(room: RoomStructure, frames: frame_lib.Frames):
  mask = room.annotated_timestamps[None, :] == frames.frame_timestamps[:, None]
  return mask.any(dim=1)
