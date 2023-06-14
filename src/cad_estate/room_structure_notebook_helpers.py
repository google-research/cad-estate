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

import colorsys
import dataclasses
import io
import re
import subprocess

import appdirs
import numpy as np
import pandas
import PIL.Image
import torch as t
import torchvision.transforms.functional as tvtF
import yt_dlp

from cad_estate import file_system as fs
from cad_estate import misc_util
from cad_estate import transformations
from cad_estate.gl import scene_renderer


@dataclasses.dataclass
class RoomStructure:
  """Describes the structural elements of a room."""

  clip_name: str
  """The clip name, format: `"{video_id}_{timestamp_first_frame}"`."""

  layout_triangles: np.ndarray
  """The geometry as a triangle soup, `float32[NUM_LAYOUT_TRI, 3, 3]`."""

  layout_triangle_flags: np.ndarray
  """Additional per-triangle flags, `float32[NUM_LAYOUT_TRI, 3, 3]`.
  Bit 1 indicates the triangle is part of a window frame. Bit 2 -- part of a
  closed door.
  """

  layout_num_tri: np.ndarray = None
  """Number of triangles for each structural element, `int64[NUM_ELEMENTS]`."""

  layout_labels: np.ndarray = None
  """The semantic labels of the structural elements, `int64[NUM_ELEMENTS]`."""


RE10K_COLS = [
    "timestamp", "fx", "fy", "cx", "cy", "z1", "z2", "e00", "e01", "e02", "e03",
    "e10", "e11", "e12", "e13", "e20", "e21", "e22", "e23"
]
"""Columns in the RealEstate10K camera frames description"""

LAYOUT_LABELS = ["<ignore>", "wall", "floor", "ceiling", "slanted"]


@dataclasses.dataclass
class VisualizerHelper:
  ffmpeg_path: str
  """Path to the FFMPEG binary."""

  cad_estate_dir: str
  """Path to the extracted RealEstate10K dataset."""

  num_frames: int
  """Number of frames to display."""

  image_width: int
  """Desired output image width."""

  scene_path: str | None = None
  """Path of the loaded scene, or None."""

  room: RoomStructure | None = None
  """The loaded scene, or None."""

  frames: t.Tensor | None = None
  """Frames from the clip of the loaded scene, or None."""

  frame_cameras: t.Tensor | None = None
  """Camera matrices corresponding to the frames above, or None."""

  def load_scene(self, re10k_path: str):
    """Loads a scene, from path to a RE10K scene file."""

    if self.scene_path == re10k_path:
      return

    print(f"Loading {re10k_path}...")

    self.scene_path = re10k_path

    # Load the video URL and the frame arguments from RE10K
    video_url, cam_info = fs.read_text(re10k_path).split("\n", 1)
    video_id = re.match(r"^https://www\.youtube\.com/watch\?v=(.+)$",
                        video_url).group(1)
    dtypes = {RE10K_COLS[0]: np.int64} | {k: np.float32 for k in RE10K_COLS[1:]}
    cam_info = pandas.read_csv(
        io.StringIO(cam_info), sep="\s+", header=None, names=RE10K_COLS,
        index_col=None, dtype=dtypes)
    first_frame_idx = cam_info.timestamp[0]

    # Read the room structure
    room_path = fs.join(self.cad_estate_dir, f"{video_id}_{first_frame_idx}",
                        "room_structure.npz")
    if not fs.exists(room_path):
      # Only ~2000 of all ~55000 clips in RE10K have layout annotations
      raise ValueError(f"No layout annotations for video '{video_url}'"
                       f" (from RE10K file '{re10k_path}').")
    self.room = RoomStructure(**np.load(io.BytesIO(fs.read_bytes(room_path))))

    # Leave only the frames that will be displayed here
    frame_idx = np.linspace(0, cam_info.shape[0] - 1, self.num_frames,
                            dtype=int)
    cam_info = cam_info.iloc[frame_idx]

    # Download the video
    video_dir = fs.join(appdirs.user_cache_dir("cad_estate"), "videos")
    video_path = fs.join(video_dir, video_id)
    fs.make_dirs(video_dir)
    if not fs.exists(video_path):
      print(f"Downloading video to '{video_path}'")
      ytdl = yt_dlp.YoutubeDL(
          params=dict(format="bv", paths=dict(home=video_dir), outtmpl=dict(
              default="%(id)s")))
      ytdl.download(video_url)
    else:
      print(f"Using downloaded video in '{video_path}'")

    # Extract the video frames
    frames_dir = fs.join(
        appdirs.user_cache_dir("cad_estate"), "frames", video_id)
    fs.make_dirs(frames_dir)
    for time_stamp in cam_info.timestamp:
      out_path = fs.join(frames_dir, f"{video_id}_{time_stamp}.png")
      if not fs.exists(out_path):
        print(f"Extracting frame into '{out_path}'")
        args = [
            self.ffmpeg_path, "-hide_banner", "-loglevel", "error", "-y", "-ss",
            str(time_stamp / 10**6), "-i", video_path, "-vframes", "1", "-f",
            "image2", out_path
        ]
        if subprocess.run(args).returncode != 0:
          raise ValueError("Unable to extract frame!")
      else:
        print(f"Using extracted frame from '{out_path}'")

    # Read the video frames and compute the cameras
    frame_list = []
    cam_list = []
    for row in cam_info.itertuples():
      frame_path = fs.join(frames_dir, f"{video_id}_{row.timestamp}.png")
      rgb = np.array(PIL.Image.open(io.BytesIO(fs.read_bytes(frame_path))))

      h, w, _ = rgb.shape
      h, w = h * self.image_width // w, self.image_width
      rgb = tvtF.convert_image_dtype(t.as_tensor(rgb), t.float32)
      rgb = tvtF.resize(rgb.permute(2, 0, 1), (h, w),
                        antialias=True).permute(1, 2, 0)
      frame_list.append(rgb)

      intrinsics = transformations.gl_projection_matrix_from_intrinsics(
          w, h, row.fx * w, row.fy * h, row.cx * w, row.cy * h)
      extrinsics = t.tensor([
          row.e00, row.e01, row.e02, row.e03, row.e10, row.e11, row.e12,
          row.e13, row.e20, row.e21, row.e22, row.e23, 0, 0, 0, 1
      ], dtype=t.float32).reshape([4, 4])
      cam_list.append(intrinsics @ extrinsics)
    self.frames = t.stack(frame_list)
    self.frame_cameras = t.stack(cam_list)

  def get_palette(self):
    color_palette = []
    for h in t.arange(0., 1., 1 / 32):
      color_palette.append(colorsys.hsv_to_rgb(h, 1, 0.7))
      color_palette.append(colorsys.hsv_to_rgb(h, 0.5, 0.7))
    color_palette = t.tensor(color_palette, dtype=t.float32)
    g = t.Generator()
    g.manual_seed(1)
    color_palette = color_palette[t.randperm(color_palette.shape[0],
                                             generator=g)]
    return color_palette

  def render_frame(self, frame_index: int):
    """Renders a single frame."""
    palette = self.get_palette()
    num_palette_colors = palette.shape[0]
    palette_darker = palette * 0.8
    palette_lighter = palette * 1.2
    palette = t.concat([palette, palette_darker, palette_lighter])
    material_ids = misc_util.dynamic_tile(self.room.layout_num_tri)
    is_window_frame = (self.room.layout_triangle_flags & 1) == 1
    is_door = (self.room.layout_triangle_flags & 2) == 2
    material_ids[is_window_frame] += 2 * num_palette_colors
    material_ids[is_door] += num_palette_colors
    _, h, w, _ = self.frames.shape
    result = scene_renderer.render_scene(  #
        self.room.layout_triangles, self.frame_cameras[frame_index], (h, w),
        diffuse_coefficients=palette, material_ids=material_ids.to(t.int32),
        cull_back_facing=False)
    return tvtF.convert_image_dtype(result, dtype=t.float32)

  def get_legend_html(self):
    html = "Legend: "
    style = ("color:black; font-weight: bold; padding: .2em; "
             "display: inline-block")
    for l, c in zip(self.room.layout_labels[1:], self.get_palette()[1:]):
      text = LAYOUT_LABELS[l]
      c = (c * 255).to(t.int32).tolist()
      html += f"<div style='{style}; background: rgb{tuple(c)};'>{text}</div>\n"
    return html
