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

import json
import pathlib
import re

import appdirs
import ipywidgets
import torch as t
import torchvision.utils as tvU

from cad_estate import download_and_extract_frames
from cad_estate import file_system as fs
from cad_estate import frames as frame_lib
from cad_estate import misc_util
from cad_estate import objects as obj_lib
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


def create_interactive_widgets(example_scenes: list[str], num_frames: int,
                               annotations_dir: str):
  """Setup the interactive widgets."""
  w_scene_name = ipywidgets.Dropdown(options=example_scenes)
  w_frame_index = ipywidgets.IntSlider(0, 0, num_frames - 1, 1)
  w_show_tracks = ipywidgets.Checkbox()

  def compute_frame_index_bounds(unused):
    if w_show_tracks.value:
      # When tracks are displayed, the set of frames matches the ones with
      # manual track completion
      frames_json = json.loads(
          fs.read_text(
              fs.join(annotations_dir, w_scene_name.value, "frames.json")))
      frames = frame_lib.load_metadata(frames_json)
      w_frame_index.max = (
          int(frames.manual_track_annotations.to(t.int64).sum()) - 1)
    else:
      w_frame_index.max = num_frames - 1

  w_show_tracks.observe(compute_frame_index_bounds)
  w_scene_name.observe(compute_frame_index_bounds)

  return w_scene_name, w_frame_index, w_show_tracks


def render_objects(objects: obj_lib.Objects, frames: frame_lib.Frames,
                   frame_index: int):
  pallette = misc_util.get_palette()
  rgb = frames.frame_images[frame_index]
  cam_mat = (
      frames.camera_intrinsics[frame_index]
      @ frames.camera_extrinsics[frame_index])
  if objects.num_tri.sum() == 0:
    synth = t.zeros_like(rgb)
  else:
    synth: t.Tensor = scene_renderer.render_scene(
        objects.triangles, cam_mat, rgb.shape[-2:], cull_back_facing=False,
        diffuse_coefficients=pallette[1:],
        material_ids=misc_util.dynamic_tile(objects.num_tri).to(t.int32))
    synth = synth.permute([2, 0, 1])

  return synth, rgb


def render_tracks(img: t.Tensor, objects: obj_lib.Objects,
                  track_boxes: t.Tensor | None, frame_index: int):
  if track_boxes is None:
    return img
  pallette = misc_util.get_palette()
  _, h, w = img.shape
  boxes = track_boxes[:, frame_index] * track_boxes.new_tensor([h, w, h, w])
  mask = ((boxes >= 0).all(dim=1) & (objects.num_tri == 0) &
          ~objects.from_automatic_track)
  boxes = boxes[mask][:, [1, 0, 3, 2]].to(t.int32)
  if not boxes.shape[0]:
    return img
  box_colors = t.cat([pallette] * 12)[mask.nonzero()[:, 0]]
  box_colors = [tuple(v) for v in (box_colors * 255).to(t.int32)]
  return tvU.draw_bounding_boxes(img, boxes, width=2, colors=box_colors)
