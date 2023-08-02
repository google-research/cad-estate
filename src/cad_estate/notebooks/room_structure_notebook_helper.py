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

import enum
import io
import pathlib
import re

import appdirs
import ipywidgets
import numpy as np
import torch as t
import torchvision.transforms.functional as tvtF

from cad_estate import download_and_extract_frames
from cad_estate import file_system as fs
from cad_estate import frames as frame_lib
from cad_estate import misc_util
from cad_estate import room_structure as struct_lib
from cad_estate.gl import scene_renderer


def download_frames(frames: frame_lib.Frames, frames_dir: str | None):
  if frames_dir:
    frames_dir = fs.abspath(frames_dir)
    return frames_dir

  video_id = re.match(r"^(.+)_\d+$", frames.clip_name).group(1)
  cache_dir = pathlib.Path(appdirs.user_cache_dir())
  frames_dir = cache_dir / "cad_estate" / "frames"
  video_path = cache_dir / "cad_estate" / "videos" / f"{video_id}.mp4"
  download_and_extract_frames.download_video(video_id, video_path)
  clip_frames_dir = frames_dir / video_path.stem
  clip_frames_dir.mkdir(parents=True, exist_ok=True)
  download_and_extract_frames.extract_frames(video_path, clip_frames_dir,
                                             frames.frame_timestamps)
  return frames_dir


def render_frame(frames: frame_lib.Frames, room: struct_lib.RoomStructure,
                 frame_index: int, width: int):
  """Renders a single frame."""
  palette = t.cat([misc_util.get_palette()] * 12)
  num_palette_colors = palette.shape[0]
  palette_darker = palette * 0.8
  palette_lighter = palette * 1.2
  palette = t.concat([palette, palette_darker, palette_lighter])
  material_ids = misc_util.dynamic_tile(room.num_tri)
  is_window_frame = (room.triangle_flags & 1) == 1
  is_door = (room.triangle_flags & 2) == 2
  material_ids[is_window_frame] += 2 * num_palette_colors
  material_ids[is_door] += num_palette_colors
  _, _, h, w = frames.frame_images.shape
  h = h * width // w
  w = width
  cam_mat = (
      frames.camera_intrinsics[frame_index]
      @ frames.camera_extrinsics[frame_index])
  synth = scene_renderer.render_scene(  #
      room.triangles, cam_mat, (h, w), diffuse_coefficients=palette,
      material_ids=material_ids.to(t.int32), cull_back_facing=False)
  synth = tvtF.convert_image_dtype(synth.permute([2, 0, 1]), t.float32)
  rgb = tvtF.resize(frames.frame_images[frame_index], (h, w), antialias=True)
  rgb = tvtF.convert_image_dtype(rgb, t.float32)
  return synth, rgb


def render_annotations(annotations: struct_lib.StructureAnnotations,
                       frame_index: int, width: int):
  _, sh, sw = annotations.structural_element_masks.shape
  h = sh * width // sw
  w = width
  palette_str = t.cat([misc_util.get_palette()] * 12)
  palette_vis = palette_str.clone()
  palette_vis[0, :] = 0
  palette_vis[1, :] = palette_vis.new_tensor([0, 0, 1.0])
  str_ann = annotations.structural_element_masks[frame_index].to(t.int64)
  vis_ann = annotations.visible_parts_masks[frame_index].to(t.int64)
  str_ann = palette_str[str_ann.reshape([-1, 1])].reshape([sh, sw, 3])
  vis_ann = palette_vis[vis_ann.reshape([-1, 1])].reshape([sh, sw, 3])
  ann_tup = [str_ann, vis_ann]
  ann_tup = [
      tvtF.resize(v.permute([2, 0, 1]), (h, w), antialias=True) for v in ann_tup
  ]
  ann_tup = [tvtF.convert_image_dtype(v, t.float32) for v in ann_tup]
  return ann_tup


def get_legend_html(room: struct_lib.RoomStructure):
  html = "Legend: "
  style = ("color:black; font-weight: bold; padding: .2em; "
           "display: inline-block")
  for l, c in zip(room.labels[1:], misc_util.get_palette()[1:]):
    text = struct_lib.STRUCTURAL_ELEMENT_TYPES[l]
    c = (c * 255).to(t.int32).tolist()
    html += f"<div style='{style}; background: rgb{tuple(c)};'>{text}</div>\n"
  return html


class AnnotationType(enum.Enum):
  """How to display the annotions."""

  # Don't display annotations
  NONE = "None",

  # Display annotations as drawn by the annotators
  RAW = "Raw",

  # Display processed annotations, where instance labels match geometry,
  # visible parts and structure are combined, and doors/windows in-painted
  PROCESSED = "Processed"


def create_interactive_widgets(example_scenes: list[str], num_frames: int,
                               annotations_dir: str):
  """Setup the interactive widgets."""
  scene_name = ipywidgets.Dropdown(options=example_scenes)
  frame_index = ipywidgets.IntSlider(0, 0, num_frames - 1, 1)
  annotation_type = ipywidgets.Dropdown(options=AnnotationType)

  def compute_frame_index_bounds(unused):
    if annotation_type.value == AnnotationType.NONE:
      frame_index.max = num_frames - 1
    else:
      # When annotations are displayed, the frames match the annotated ones
      room_npz = fs.read_bytes(
          fs.join(annotations_dir, scene_name.value, "room_structure.npz"))
      room = struct_lib.load_room_structure(np.load(io.BytesIO(room_npz)))
      frame_index.max = room.annotated_timestamps.shape[0] - 1

  annotation_type.observe(compute_frame_index_bounds)
  scene_name.observe(compute_frame_index_bounds)

  return scene_name, frame_index, annotation_type
