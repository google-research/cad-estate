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
"""Miscellaneous utilities that don't fit anywhere else."""

from typing import Iterable, Optional, Union

import numpy as np
import torch as t

InputTensor = Union[t.Tensor, np.ndarray, int, float, Iterable]


def to_tensor(v: InputTensor, dtype: t.dtype,
              device: Optional[Union[t.device, str]] = None) -> t.Tensor:
  """Converts a value to tensor, checking the type.

  Args:
    v: The value to convert. If it is already a tensor or an array, this
      function checks that the type is equal to dtype. Otherwise, uses
      torch.as_tensor to convert it to tensor.
    dtype: The required type.
    device: The target tensor device (optional(.

  Returns:
    The resulting tensor

  """
  if not t.is_tensor(v):
    if hasattr(v, "__array_interface__"):
      # Preserve the types of arrays. The must match the given type.
      v = t.as_tensor(v)
    else:
      v = t.as_tensor(v, dtype=dtype)

  if v.dtype != dtype:
    raise ValueError(f"Expecting type '{dtype}', found '{v.dtype}'")

  if device is not None:
    v = v.to(device)

  return v


def dynamic_tile(partition_lengths: t.Tensor) -> t.Tensor:
  """Computes dynamic tiling with the given partition lengths.

  Args:
    partition_lengths: The partition lengths, int64[num_partitions]

  Returns:
    A 1D int tensor, containing  partition_lengths[0] zeros,
    followed by partition_lengths[1] ones, followed by
    partition_lengths[2] twos, and so on.
  """
  partition_lengths = t.as_tensor(partition_lengths)
  non_zero_idx = partition_lengths.nonzero()[:, 0]
  partition_lengths = partition_lengths[non_zero_idx]
  start_index = partition_lengths.cumsum(0)
  if start_index.shape == (0,):
    return start_index
  start_index, num_elements = start_index[:-1], start_index[-1]
  result: t.Tensor = partition_lengths.new_zeros([num_elements.item()])
  start_index = start_index[start_index < num_elements]
  result[start_index] = 1
  result = result.cumsum(0, dtype=partition_lengths.dtype)
  return non_zero_idx[result]
