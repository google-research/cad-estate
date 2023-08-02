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
"""Library for loading and manipulating video frames."""

import dataclasses
import io
import re
import PIL.Image

import numpy as np
import torch as t

from cad_estate import file_system as fs
from cad_estate import misc_util
from cad_estate import input_file_structures as input_struct_lib
from cad_estate import transformations as transform_lib


async def read_and_decode_image(image_path: str):
  """Asynchronously reads and decodes an image."""
  image_bytes = await fs.read_bytes_async(image_path)
  image = PIL.Image.open(io.BytesIO(image_bytes))
  image = t.as_tensor(np.array(image), dtype=t.uint8)
  if len(image.shape) == 3:
    if image.shape[-1] != 3:
      raise ValueError("Only RGB and GrayScale images supported!")
    image = image.permute([2, 0, 1])
  elif len(image.shape) != 2:
    raise ValueError("Only RGB and GrayScale images supported!")
  return image


# We use the following symbols to describe tensors below:
# IH, IW     : Image height and width
# NUM_FRAMES : Number of loaded frames in a video


@dataclasses.dataclass
class Frames(misc_util.TensorContainerMixin):
  """Contains frames of a video."""

  clip_name: str
  """The RealEstate-10K clip name.
  Uniquely identifies the scene. Consists of a YouTube video ID,
  followed by "_", and then by a start timestamp.
  """

  frame_timestamps: t.Tensor
  """Frame timestamps (microseconds since video start), `int[NUM_FRAMES]`."""

  frame_images: t.Tensor | None
  """The frame images, `uint8[NUM_FRAMES, 3, IH, IW]`.
  None, if the frames are not loaded yet.
  """

  camera_intrinsics: t.Tensor
  """Camera intrinsics (view->screen transform), `float32[NUM_FRAMES, 4, 4]`.
  Entries correspond to frames.
  """

  camera_extrinsics: t.Tensor
  """Camera extrinsics (world->view transform), `float32[NUM_FRAMES, 4, 4]`.
  Entries correspond to frames.
  """

  manual_track_annotations: t.Tensor
  """Whether tracks were completed manually on a frame, `bool[NUM_FRAMES]`."""


def load_metadata(frames_json: input_struct_lib.FramesFile, z_near=0.1,
                  z_far=200.0):
  """Loads the frames metadata."""

  clip_name = frames_json["clip_name"]
  frame_timestamps, manual_track_annotations = [], []
  camera_intrinsics, camera_t, camera_r = [], [], []
  for frame in frames_json["frames"]:
    frame_timestamps.append(frame["timestamp"])
    camera_intrinsics.append(frame["intrinsics"])
    camera_t.append(frame["extrinsics"]["translation"])
    camera_r.append(frame["extrinsics"]["rotation"])
    manual_track_annotations.append(frame["key_frame"])

  h, w = frames_json["image_size"]
  fx, fy, cx, cy = t.as_tensor(camera_intrinsics, dtype=t.float32).unbind(-1)
  camera_intrinsics = transform_lib.gl_projection_matrix_from_intrinsics(
      w, h, fx, fy, cx, cy, z_near, z_far)

  camera_extrinsics = transform_lib.chain([
      transform_lib.translate(camera_t),
      transform_lib.quaternion_to_rotation_matrix(camera_r)
  ])

  frame_timestamps = t.as_tensor(frame_timestamps)
  manual_track_annotations = t.as_tensor(manual_track_annotations)

  return Frames(clip_name=clip_name, frame_timestamps=frame_timestamps,
                frame_images=None, camera_intrinsics=camera_intrinsics,
                camera_extrinsics=camera_extrinsics,
                manual_track_annotations=manual_track_annotations)


async def load_images(frames: Frames, frames_root_dir: str) -> Frames:
  """Reads the frame images in parallel (using async)."""

  video_name = re.match(r"^(.+)_\d+$", frames.clip_name).group(1)
  image_paths = [
      fs.join(frames_root_dir, video_name, f"{video_name}_{v}.jpg")
      for v in frames.frame_timestamps.cpu()
  ]

  # Load images in parallel to mask latencies
  images = await fs.await_in_parallel(
      [read_and_decode_image(v) for v in image_paths])
  images = t.stack(images, dim=0)

  return dataclasses.replace(frames, frame_images=images)


def filter(frames: Frames, keep: t.Tensor) -> Frames:
  """Filters the frames and their metadata.

  Args:
    frames: The input frames
    keep: Which frames to keep. Either a boolean mask (`bool[NUM_FRAMES]`) or a
      list of indices (`int64[NUM_FRAMES_TO_KEEP]`).

  Returns:
    The frames object, with frames filtered according to the arguments.
  """
  frame_images = frames.frame_images
  if frame_images is not None:
    frame_images = frame_images[keep]
  return Frames(
      clip_name=frames.clip_name,
      frame_timestamps=frames.frame_timestamps[keep],
      frame_images=frame_images,
      camera_intrinsics=frames.camera_intrinsics[keep],
      camera_extrinsics=frames.camera_extrinsics[keep],
      manual_track_annotations=frames.manual_track_annotations[keep],
  )


def _regular_indices_impl(frames: Frames, num_frames_to_keep: int,
                          offset: t.Tensor) -> t.Tensor:
  """Returns evenly spaced frame indices."""
  device = frames.frame_timestamps.device
  frame_index = t.arange(num_frames_to_keep, dtype=t.float32, device=device)
  if offset is not None:
    frame_index = frame_index + offset
  num_frames = len(frames.frame_timestamps)
  frame_index = frame_index / num_frames_to_keep * num_frames
  frame_index = frame_index.floor().clip(0, num_frames - 1).to(t.int64)
  return frame_index


def sample_stratified(frames: Frames, num_frames_to_keep: int,
                      rng: t.Generator | None = None) -> t.Tensor:
  """Samples frame indices in a stratified manned, returns boolean mask."""
  offset = t.rand(num_frames_to_keep, generator=rng,
                  device=frames.frame_timestamps.device)
  return _regular_indices_impl(frames, num_frames_to_keep, offset)


def sample_regular(frames: Frames, num_frames_to_keep: int) -> t.Tensor:
  """Samples evenly spaced frame indices, returns boolean mask."""
  return _regular_indices_impl(frames, num_frames_to_keep, None)
