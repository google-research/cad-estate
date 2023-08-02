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
"""Dataset classes for reading CAD-Estate annotations and their videos.

Example usage during training:
```python
from cad_estate import objects as obj_lib
from cad_estate import datasets as dataset_lib

cfg = datasets.ObjectDatasetConfig(
    split_file_path=fs.abspath("~/prj/cad_estate/data/obj_test.txt"),
    annotation_directory=fs.abspath("~/prj/cad_estate/data/annotations"),
    frames_directory=fs.abspath("~/prj/cad_estate/data/frames"), num_frames=12)
shapenet_meta = obj_lib.load_shapenet_metadata(
    fs.abspath("~/prj/cad_estate/data/shapenet_npz"))

for epoch in range(num_epochs):
  dataset = ObjectDataset(cfg, shapenet_meta, seed=epoch)
  data_loader = DataLoader(ObjectDataset(config, shapenet_meta, epoch))
  for batch in data_loader:
    ...
```

Example usage during evaluation:

```python
dataset = ObjectDataset(cfg, shapenet_meta, seed=None)
data_loader = DataLoader(ObjectDataset(config, shapenet_meta, epoch))
for batch in data_loader:
  ...
```

"""

import asyncio
import dataclasses
import io
import json
import math
import re
from typing import Generic, TypeVar

import numpy as np
import torch as t
import torch.utils.data

from cad_estate import file_system as fs
from cad_estate import frames as frame_lib
from cad_estate import objects as obj_lib
from cad_estate import room_structure as struct_lib


@dataclasses.dataclass
class DatasetCommonConfig:
  """Configures the common part between structure and object datasets."""

  split_file_path: str
  """Path to a text file containing the scene names that are part of the
  dataset split (one scene per line)."""

  annotation_directory: str
  """Path to directory containing the annotations."""

  frames_directory: str
  """Path to directory containing the frame images."""

  num_frames: int
  """Number of frames to sample for a clip."""

  replication_factor: float = 1.0
  """Allows replicating/trimming the dataset."""

  znear: float = 0.1
  """ZNear, used to compute an OpenGL compatible projection matrix."""

  zfar: float = 50
  """ZFar, used to compute an OpenGL compatible projection matrix."""


@dataclasses.dataclass
class ObjectDatasetConfig(DatasetCommonConfig):
  """Configures an object dataset."""

  read_tracks: bool = False
  """Whether to read the 2D object tracks."""

  only_frames_with_manual_tracks: bool = False
  """If there are frames with manual track completion, sample only from them."""


@dataclasses.dataclass
class RoomStructureDatasetConfig(DatasetCommonConfig):
  """Configures a room structures dataset."""

  load_raw_annotations: bool | None = None
  """Whether to load raw or processed annotations.
  If None, annotations are not loaded."""

  only_frames_with_annotations: bool = False
  """Whether to sample only from frames with annotations."""


@dataclasses.dataclass
class ObjectsElement:
  """Element returned when iterating an objects dataset."""

  frames: frame_lib.Frames
  """The loaded frames."""

  objects: obj_lib.Objects
  """The loaded objects."""

  track_boxes: t.Tensor | None
  """The track boxes, `float32[NUM_OBJ, NUM_FRAMES, 4]`. See the
  `cad_estate.objects.load_track_boxes` for more details."""


@dataclasses.dataclass
class RoomStructuresElement:
  """Element returned when iterating a room structure dataset."""

  frames: frame_lib.Frames
  """The loaded frames."""

  room: struct_lib.RoomStructure
  """The loaded room structure."""

  annotations: struct_lib.StructureAnnotations | None
  """The room structure annotations."""


_DATASET_CACHE: dict[str, str] = {}  # t[str, str] = {}


def _read_cached(path: str, cache_enabled: bool) -> str:
  if path not in _DATASET_CACHE or not cache_enabled:
    _DATASET_CACHE[path] = fs.read_text(path)
  return _DATASET_CACHE[path]


Config = TypeVar("Config", bound=DatasetCommonConfig)


class DatasetBase(Generic[Config]):
  """Contains the common functionality between objects and room structures."""

  def __init__(self, config: Config, seed: int | None,
               cache_dataset_description: bool):
    """Initializes the dataset.

    Args:
      config: The dataset config
      seed: Seed for sampling frames and shuffling scenes (see below).
      cache_dataset_description: Whether to cache the contents of
        `config.split_file_path` in memory, for faster re-initialization.

    The scene iteration order and the frames sampled for a scene, depend only
    on the construction arguments (configuration and seed). Iterating over two
    datasets with identical construction arguments, or iterating over a dataset
    multiple times will yield the exact same results.

    To sample different frames and to visit the scenes in a different order
    during training, create a new dataset each for each epoch:
    ```
    for epoch in range(num_epochs):
      data_loader = DataLoader(ObjectDataset(config, shapenet_meta, epoch))
      for batch in data_loader:
        ...
    ```

    Frames are chosen by sampling their indices in a stratified manner.
    If `seed` is `None`, the indices are evenly spaced.

    The `_compute_indices` and `_sample_frames` methods can be overriden for
    a different shuffling and sampling behavior.
    """

    self.config: Config = config
    split_description = _read_cached(self.config.split_file_path,
                                     cache_dataset_description)
    self.original_scene_names = self._read_scene_names(split_description)
    self.seed = seed
    self.indices = self._compute_indices()

  def _read_scene_names(self, split_description: str):
    result = [v for v in split_description.splitlines() if v]
    result = [v for v in result if not re.match(r"^(\s*#.*|\s*)$", v)]
    return np.array(result, dtype=np.str_)

  def _compute_indices(self) -> t.Tensor:
    num_scenes = len(self.original_scene_names)
    if self.seed is None:
      _i = lambda: t.arange(num_scenes, device="cpu")
    else:
      g = t.Generator()
      g.manual_seed(self.seed)
      _i = lambda: t.randperm(num_scenes, generator=g, device="cpu")
    max_repeats = int(math.ceil(self.config.replication_factor))
    indices = t.cat([_i() for _ in range(max_repeats)])
    indices_len = int(num_scenes * self.config.replication_factor)
    indices = indices[:indices_len]

    return indices

  @property
  def scene_names(self):
    """Returns the already shuffled scene names."""
    return self.original_scene_names[self.indices]

  def __len__(self):
    return self.indices.shape[0]

  def _sample_frames(self, frames: frame_lib.Frames):
    if self.seed is None:
      return frame_lib.sample_regular(frames, self.config.num_frames)
    g = t.Generator()
    g.manual_seed(int(self.indices[self.seed]))
    return frame_lib.sample_stratified(frames, self.config.num_frames, g)

  def _get_frame_metadata(self, index: int):
    scene_name = self.scene_names[index]
    json_path = fs.join(self.config.annotation_directory, scene_name,
                        "frames.json")
    frames = frame_lib.load_metadata(json.loads(fs.read_text(json_path)))
    return frames


class ObjectDataset(DatasetBase[ObjectDatasetConfig],
                    torch.utils.data.Dataset[ObjectsElement]):

  def __init__(self, config: ObjectDatasetConfig,
               shapenet_meta: obj_lib.ShapeNetMetadata, seed: int | None,
               cache_dataset_description: bool = True):
    super().__init__(config, seed, cache_dataset_description)
    self.shapenet_meta = shapenet_meta

  def __getitem__(self, index: int) -> ObjectsElement:
    scene_name = self.scene_names[index]
    json_path = fs.join(self.config.annotation_directory, scene_name,
                        "objects.json")
    obj_json = json.loads(fs.read_text(json_path))
    objects = asyncio.run(obj_lib.load_objects(obj_json, self.shapenet_meta))

    frames = self._get_frame_metadata(index)
    if (self.config.only_frames_with_manual_tracks
        and frames.manual_track_annotations.any()):
      frames = frame_lib.filter(frames, frames.manual_track_annotations)
    frames = frame_lib.filter(frames, self._sample_frames(frames))
    frames = asyncio.run(
        frame_lib.load_images(frames, self.config.frames_directory))

    tracks = None
    if self.config.read_tracks:
      tracks = obj_lib.load_track_boxes(obj_json, frames)

    return ObjectsElement(frames, objects, tracks)


class RoomStructuresDataset(
    DatasetBase[RoomStructureDatasetConfig],
    torch.utils.data.Dataset[RoomStructureDatasetConfig],
):

  def __init__(self, config: ObjectDatasetConfig, seed: int | None,
               cache_dataset_description: bool = True):
    super().__init__(config, seed, cache_dataset_description)

  def __getitem__(self, index: int) -> RoomStructuresElement:
    scene_name = self.scene_names[index]
    room_bytes = fs.read_bytes(
        fs.join(self.config.annotation_directory, scene_name,
                "room_structure.npz"))
    room = struct_lib.load_room_structure(np.load(io.BytesIO(room_bytes)))

    frames = self._get_frame_metadata(index)
    if self.config.only_frames_with_annotations:
      mask = struct_lib.annotated_frames_mask(room, frames)
      frames = frame_lib.filter(frames, mask)
    frames = frame_lib.filter(frames, self._sample_frames(frames))
    frames = asyncio.run(
        frame_lib.load_images(frames, self.config.frames_directory))

    annotations = None
    if self.config.load_raw_annotations is not None:
      annotations = asyncio.run(
          struct_lib.load_annotations(room, frames,
                                      self.config.annotation_directory,
                                      self.config.load_raw_annotations))

    return RoomStructuresElement(frames, room, annotations)
