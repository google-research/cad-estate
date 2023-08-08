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
"""Downloads CAD-estate videos and extracts their frames."""

import asyncio
import collections
import contextlib
import dataclasses
import datetime
import functools
import json
import logging
import pathlib
import re
import shutil
import subprocess
import tempfile
from concurrent import futures
from typing import Iterable

import appdirs
import numpy as np
import pandas
import tqdm
import yt_dlp

from cad_estate import structured_arg_parser as arg_lib

log = logging.getLogger(__name__)
log_ytdlp = logging.getLogger(yt_dlp.__name__)
FFMPEG_PATH = "/usr/bin/ffmpeg"


def download_video(video_id: str, out_path: pathlib.Path | None = None):
  video_url = f"https://www.youtube.com/watch?v={video_id}"
  if not out_path:
    out_path = (
        pathlib.Path(appdirs.user_cache_dir("cad_estate")) / "videos" /
        f"{video_id}.mp4")
  out_path.parent.mkdir(parents=True, exist_ok=True)
  if not out_path.exists():
    log.info(f"Downloading video to '{out_path}'")
    ytdl = yt_dlp.YoutubeDL(
        params=dict(format="bv", logger=log_ytdlp, paths=dict(
            home=str(out_path.parent)), outtmpl=dict(default=out_path.name)))
    ytdl.download(video_url)
  else:
    log.info(f"Using previously downloaded video in '{out_path}'")

  return out_path


def extract_frames(video_path: pathlib.Path, out_dir: pathlib.Path,
                   frame_timestamps: Iterable[int],
                   tmp_dir: pathlib.Path | None = None,
                   ffmpeg_path=FFMPEG_PATH):
  frame_discrepancy_path = out_dir / "frame_discrepancy.json"
  if frame_discrepancy_path.exists():
    return json.loads(frame_discrepancy_path.read_text())
  if not tmp_dir:
    tmp_dir_ctx = tempfile.TemporaryDirectory()
    tmp_dir = pathlib.Path(tmp_dir_ctx.name)
  else:
    tmp_dir_ctx = contextlib.nullcontext()

  with tmp_dir_ctx:
    # Extract the video frames
    assert tmp_dir.exists()
    if any(tmp_dir.iterdir()):
      raise ValueError("Unpack directory must be empty!")

    args = [
        ffmpeg_path, "-hide_banner", "-loglevel", "error", "-y", "-vsync",
        "vfr", "-i", video_path, "-frame_pts", "1", "-r", "1000000", "-f",
        "image2", "-qscale:v", "2", f"{tmp_dir}/out%d.jpg"
    ]

    log.info(f"Extracting frames of '{video_path}', using:\n", " ".join(args))
    ff_proc = subprocess.run(args, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True)
    if ff_proc.returncode != 0:
      raise ValueError(f"ffmpeg failed to extract frames:\n{ff_proc.stdout}")

    # Match disk timestamps to annotation timestamps
    tmp_dir.parent.mkdir(parents=True, exist_ok=True)
    jpg_files = tmp_dir.glob("out*.jpg")
    disk_timestamps = sorted(
        [int(v.stem.removeprefix("out")) for v in jpg_files])
    disk_timestamps = np.array(disk_timestamps, np.int64)
    frame_timestamps = np.array(frame_timestamps, np.int64)
    idx = np.searchsorted(disk_timestamps, frame_timestamps, side="right")
    idx1 = np.clip(idx - 1, 0, len(disk_timestamps) - 1)
    idx2 = np.clip(idx, 0, len(disk_timestamps) - 1)
    dist1 = np.abs(disk_timestamps[idx1] - frame_timestamps)
    dist2 = np.abs(disk_timestamps[idx2] - frame_timestamps)
    found = disk_timestamps[np.where(dist1 < dist2, idx1, idx2)]

    frame_discrepancy = np.abs(found - frame_timestamps)
    frame_discrepancy_stats = {
        "min": float(frame_discrepancy.min()),
        "max": float(frame_discrepancy.max()),
        "mean": float(frame_discrepancy.mean()),
        "std": float(frame_discrepancy.std())
    }

    src_paths = [tmp_dir / f"out{v}.jpg" for v in found]
    dst_paths = [
        out_dir / f"{video_path.stem}_{v}.jpg" for v in frame_timestamps
    ]

    for src_path, dst_path in zip(src_paths, dst_paths, strict=True):
      shutil.copy(src_path, dst_path)

    frame_discrepancy_path.write_text(json.dumps(frame_discrepancy_stats))

    return frame_discrepancy_stats


def get_video_timestamps(json_path: pathlib.Path):
  ann_json = json.loads(json_path.read_text())
  clip_name = ann_json["clip_name"]
  video_id = re.match(r"^(.+)_\d+$", clip_name).group(1)
  timestamps = [v["timestamp"] for v in ann_json["frames"]]
  return video_id, timestamps


@dataclasses.dataclass
class Args:
  cad_estate_dir: str = arg_lib.flag("Root directory of CAD estate.")
  skip_download: bool = arg_lib.flag("Skip the video download step.",
                                     default=False)
  skip_extract: bool = arg_lib.flag("Skip the frame extraction step.",
                                    default=False)
  parallel_extract_tasks: int = arg_lib.flag(
      "How many frame extraction tasks to run in parallel", default=1)
  debug_run: bool = arg_lib.flag("Run in debug mode (process less videos).",
                                 default=False)


class CadEstateDirs:

  def __init__(self, root_dir: str | pathlib.Path):
    self.root = pathlib.Path(root_dir)
    self.annotations = self.root / "annotations"
    self.raw_videos = self.root / "raw_videos"
    self.frames = self.root / "frames"


def download_video_and_catch_errors(video_id: str, args: Args):
  try:
    dirs = CadEstateDirs(args.cad_estate_dir)
    video_path = dirs.raw_videos / f"{video_id}.mp4"
    download_video(video_id, video_path)
    return (video_id, True, "Download successful.")
  except Exception as e:
    return video_id, False, "; ".join(e.args)


def extract_frames_and_catch_errors(video_id: str, timestamps: list[int],
                                    args: Args):
  try:
    dirs = CadEstateDirs(args.cad_estate_dir)
    video_path = dirs.raw_videos / f"{video_id}.mp4"
    out_dir = dirs.frames / video_path.stem
    out_dir.mkdir(exist_ok=True, parents=True)
    frame_stats = extract_frames(video_path, out_dir, timestamps)
    return (video_id, True, frame_stats["max"], "Timestamp discrepancy (µs): " +
            " ".join(f"{k}={float(v):.0f}" for k, v in frame_stats.items()))
  except Exception as e:
    return video_id, False, -1, "; ".join(e.args)


async def extract_frames_in_parallel(frames_of_interest: dict[str, list[int]],
                                     args: Args):

  extract_results = []
  with futures.ThreadPoolExecutor(
      max_workers=args.parallel_extract_tasks) as pool:
    loop = asyncio.get_running_loop()
    tasks = [
        loop.run_in_executor(
            pool,
            functools.partial(extract_frames_and_catch_errors, video_id,
                              time_stamps, args))
        for video_id, time_stamps in frames_of_interest.items()
    ]
    iter_with_progress = tqdm.tqdm(
        asyncio.as_completed(tasks), "Extracting frames",
        total=len(frames_of_interest))
    for f in iter_with_progress:
      extract_results.append(await f)
  return extract_results


def main():
  args = arg_lib.parse_flags(Args)
  dirs = CadEstateDirs(args.cad_estate_dir)

  now = datetime.datetime.now()
  log_path = dirs.root / f"log_{now:%y_%m_%d__%H_%M}.txt"
  logging.basicConfig(filename=str(log_path), level=logging.INFO)

  annotation_paths = sorted(dirs.annotations.glob("*/frames.json"))
  if args.debug_run:
    annotation_paths = annotation_paths[:50]

  video_timestamps = [
      get_video_timestamps(v)
      for v in tqdm.tqdm(annotation_paths, "Loading frame timestamps")
  ]
  frames_of_interest = collections.defaultdict(lambda: set())
  for video_id, timestamps in video_timestamps:
    frames_of_interest[video_id] |= set(timestamps)
  frames_of_interest = {k: sorted(v) for k, v in frames_of_interest.items()}

  if not args.skip_download:
    download_results = [
        download_video_and_catch_errors(video_id, args)
        for video_id in tqdm.tqdm(frames_of_interest, "Downloading videos")
    ]
    df = pandas.DataFrame(download_results,
                          columns=["video_id", "is_successful", "log"])
    csv_path = dirs.root / f"download_results_{now:%y_%m_%d__%H_%M}.csv"
    csv_path.write_text(df.set_index("video_id").to_csv())
    print(f"Downloaded {df.shape[0]} videos, "
          f"with {(~df.is_successful).sum()} errors.")

  if not args.skip_extract:
    extract_results = asyncio.run(
        extract_frames_in_parallel(frames_of_interest, args))
    df = pandas.DataFrame(
        extract_results,
        columns=["video_id", "is_successful", "max_frame_discrepancy", "log"])
    df.sort_values("video_id", inplace=True)

    csv_path = dirs.root / f"process_results_{now:%y_%m_%d__%H_%M}.csv"
    csv_path.write_text(df.set_index("video_id").to_csv())

    downloaded_videos: set[str] = set(df[df.is_successful].video_id.tolist())
    scene_list = sorted([
        v.parent.name
        for v in annotation_paths
        if re.match(r"^(.+)_\d+$", v.parent.name).group(1) in downloaded_videos
    ])
    scene_list_path = dirs.root / f"scene_list_{now:%y_%m_%d__%H_%M}.txt"
    scene_list_path.write_text("\n".join(scene_list))

    print(f"Extracted frames for {df.shape[0]} videos, "
          f"with {(~df.is_successful).sum()} errors.")


if __name__ == "__main__":
  main()
