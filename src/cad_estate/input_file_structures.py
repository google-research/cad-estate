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
"""Type annotations for the input files structures."""

from typing import TypedDict

import numpy as np

# Camera extrinsics, consisting of rotation, followed by translation.
# Used in `Frame` below.
CameraExtrinsics = TypedDict(
    "CameraExtrinsics", {
        # 3D translation `(x, y, z)`
        "translation": tuple[float, float, float],

        # Rotation quaternion `(w, x, y, z)`
        "rotation": tuple[float, float, float, float]
    })

# A single frame annotation, used in `FramesFile` below
Frame = TypedDict(
    "Frame", {
        # Frame timestamp
        "timestamp": int,

        # Whether tracks were completed manually on this frame.
        "key_frame": bool,

        # The camera intrinsics `(fx, fy, cx, cy)`.
        "intrinsics": tuple[float, float, float, float],

        # The camera extrinsics.
        "extrinsics": CameraExtrinsics
    })

# A single annotated point correspondence, used in `AnnotatedObject` below.
PointCorrespondence = TypedDict("PointCorrespondence", {

    # 2D point on the video frame (normalized)
    "2d": tuple[float, float],

    # 3D point on the CAD model (in object space)
    "3d": tuple[float, float, float]
})

# Track annotations on a frame, used in `AnnotatedObject` below.
TrackEntry = TypedDict(
    "TrackEntry", {

        # The frame timestamp
        "timestamp": int,

        # The track box, in normalized coordinates
        "box": tuple[float, float, float, float],

        # The annotated 2D <=> 3D correspondences
        "correspondences": list[PointCorrespondence] | None
    })

# An annotated object
AnnotatedObject = TypedDict(
    "AnnotatedObject", {
        # Unique track/object ID
        "track_id": str,

        # Whether the object track was annotated automatically on manually
        "is_track_automatic": bool,

        # ShapeNet model ID. Can be absent, if the annotation pipeline failed
        # to propose relevant 3D models to the annotator.
        "cad_id": str,

        # ShapeNet class of the track
        "class": str,

        # Object offset in world space `(x, y, z)`. Absent, if the
        # aligned 3D object did not pass verification. Order of transformations:
        # mirror => scale => rotation => translation
        "translation": tuple[float, float, float],

        # The rotation quaternion in object space, `(w, x, y, z)`. Absent if
        # the aligned 3D object did not pass verification.
        "rotation": tuple[float, float, float, float],

        # Scale in object space, `(sx, sy, sz)`. Absent if the
        # aligned 3D object did not pass verification.
        "scale": tuple[float, float, float],

        # Whether the object needs to be mirrored in object space
        "is_mirrored": bool,

        # Track annotations
        "track": list[TrackEntry]
    })

# Describes the structure of an `objects.json` file.
ObjectsFile = TypedDict("ObjectsFile", {
    "clip_name": str,
    "objects": list[AnnotatedObject]
})

# Describes the structure of a `frames.json` file.
FramesFile = TypedDict(
    "FramesFile", {
        # The clip name
        "clip_name": str,

        # Image size, `(height, width)`
        "image_size": tuple[float, float] | None,

        # The frame annotations
        "frames": list[Frame],
    })


# Describes the structure of a `room_structure.npz` file.
RoomStructureFile = TypedDict(
    "RoomStructureFile", {
        # The clip name, `str[]`
        "clip_name": np.ndarray,

        # Structural element triangles, `float32[NUM_STRUCT_TRI, 3, 3]`.
        # Triangles of each structural element are stored sequentially.
        "layout_triangles": np.ndarray,

        # Additional per-triangle flags, `int64[NUM_STRUCT_TRI, 3]`.
        # Bit 1 indicates the triangle is part of a window frame.
        # Bit 2 -- part of a closed door.
        "layout_triangle_flags": np.ndarray,

        # Number of triangles for each structural element,
        # `int64[NUM_STRUCT_ELEM]`. The first `num_tri[0]` triangles in
        # `triangles` belong to the first structural element, the next
        # `num_tri[1]` -- to the second, and so on.
        "layout_num_tri": np.ndarray,

        # Semantic labels of the structural elements, `int64[NUM_STRUCT_ELEM]`.
        # STRUCTURAL_ELEMENT_TYPES maps these to class names.
        "layout_labels": np.ndarray,

        #Timestamps of the annotated frames, `int64[NUM_ANN_FRAMES]`.
        "annotated_timestamps": np.ndarray,
    })
