# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: spopov@google.com (Stefan Popov)
#
"""Converts ShapeNet CAD models to binary format."""

import dataclasses
import logging

import io
import numpy as np
import ray
import tqdm

from cad_estate import structured_arg_parser as arg_lib
from cad_estate import file_system as fs

log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class Args:
  """Converts ShapeNet CAD models to binary format."""
  shapenet_root: str = arg_lib.flag("Path to ShapeNet's root directory.")
  output_root: str = arg_lib.flag("Path to the output root directory.")


def read_obj(opj_path: str):
  """A simple OBJ file reader."""
  vertices = []
  faces = []
  for line in fs.read_text(opj_path).split("\n"):
    parts = line.strip().split()
    if not parts:
      continue

    if parts[0] == 'v':
      vertices.append([float(v) for v in parts[1:4]])

    if parts[0] == 'f':
      faces.append([int(p.split('/')[0]) - 1 for p in parts[1:4]])

  vertices = np.array(vertices, np.float32)
  faces = np.array(faces, np.int32)
  return vertices[faces]


def cleanup_mesh(mesh: np.ndarray):
  """Removes degenerate triangles from a mesh."""
  s1 = mesh[:, 2] - mesh[:, 0]
  s2 = mesh[:, 1] - mesh[:, 0]
  l1 = np.linalg.norm(s1, axis=-1)
  l2 = np.linalg.norm(s2, axis=-1)
  eps = 1e-27
  is_degenerate = (l1 < eps) | (l2 < eps)
  l1 = np.maximum(l1, eps)
  l2 = np.maximum(l1, eps)

  s1 /= l1[..., None]
  s2 /= l2[..., None]
  g = np.cross(s1, s2, axis=-1)
  lg = np.linalg.norm(g, axis=-1)

  is_degenerate |= lg < 1e-10

  keep_indices, = np.where(~is_degenerate)
  mesh = mesh[keep_indices]

  return mesh


def process_mesh(input_path: str, output_root: str):
  log.info(f"Processing {input_path}...")
  fn_parts = fs.splitall(input_path)
  label = fn_parts[-4]
  mesh_id = fn_parts[-3]

  mesh = read_obj(input_path)
  mesh = cleanup_mesh(mesh)

  npz_path = fs.join(output_root, label, mesh_id + ".npz")

  np.savez_compressed(fl := io.BytesIO(), vertices=mesh, label=label,
                      mesh_id=mesh_id)
  fs.make_dirs(fs.dirname(npz_path))
  fs.write_bytes(npz_path, fl.getvalue())


def main():
  args = arg_lib.parse_flags(Args)

  sn_root_dir = fs.normpath(fs.abspath(args.shapenet_root))
  print("Reading mesh file names ...")
  obj_files = sorted(
      fs.glob_pattern(fs.join(sn_root_dir, "*/*/models/model_normalized.obj")))

  out_dir = fs.normpath(fs.abspath(args.output_root))

  print(f"Converting {len(obj_files)} meshes from {sn_root_dir} to {out_dir}")

  ray.init()
  process_fn = ray.remote(process_mesh)
  tasks = [process_fn.remote(v, out_dir) for v in obj_files]

  progress_bar = tqdm.tqdm(total=len(tasks))
  while tasks:
    done, tasks = ray.wait(tasks, num_returns=len(tasks), timeout=0.3)
    progress_bar.update(len(done))


if __name__ == '__main__':
  main()
