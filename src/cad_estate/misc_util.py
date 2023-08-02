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
"""Miscellaneous utilities that don't fit anywhere else."""

import colorsys
import dataclasses
import re
from typing import Any, Callable, Iterable, TypeVar

import numpy as np
import torch as t

InputTensor = t.Tensor | np.ndarray | int | float | Iterable
TorchDevice = t.device | str | None
T = TypeVar("T")


class TensorContainerMixin:
  """Allows unified operation on all tensors contained in a dataclass."""

  def _apply(self, fn: Callable[[t.Tensor], t.Tensor]):
    result = []
    for field in dataclasses.fields(self):
      field = getattr(self, field.name)
      if t.is_tensor(field):
        field = fn(field)
      elif isinstance(field, list) or isinstance(field, tuple):
        field = [fn(e) if t.is_tensor(e) else e for e in field]
      elif isinstance(field, TensorContainerMixin):
        field = field._apply(fn)
      result.append(field)
    return type(self)(*result)

  def cuda(self: T) -> T:
    return self._apply(lambda v: v.cuda())

  def cpu(self: T) -> T:
    return self._apply(lambda v: v.cpu())

  def detach(self: T) -> T:
    return self._apply(lambda v: v.detach())

  def numpy(self: T) -> T:
    return self._apply(lambda v: v.numpy())

  def to(self: T, device: TorchDevice) -> T:
    return self._apply(lambda v: v.to(device))

  def __getitem__(self: T, index: Any) -> T:
    return self._apply(lambda v: v[index])

  def get_structure(self) -> dict[str, str]:
    """Debugging routine, returns type and shape of the each field."""
    result = {}
    for field in dataclasses.fields(self):
      v = getattr(self, field.name)
      if t.is_tensor(v):
        v: t.Tensor = v.detach()
        dtype = re.sub(r"^torch\.", "", str(v.dtype))
        structure = f"t.{dtype}{list(v.shape)}({v.device})"
      elif isinstance(v, np.ndarray):
        structure = f"np.{v.dtype.name}{list(v.shape)}"
      elif isinstance(v, list):
        structure = f"list[{len(v)}]"
      elif isinstance(v, tuple):
        structure = f"tuple[{len(v)}]"
      else:
        structure = f"{type(v).__name__}"
      result[field.name] = structure
    return result


def to_tensor(v: InputTensor, dtype: t.dtype,
              device: t.device | str | None = None) -> t.Tensor:
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


def get_palette():
  """Creates a color palette with 32 entries."""
  color_palette = []
  for h in t.arange(0., 1., 1 / 32):
    color_palette.append(colorsys.hsv_to_rgb(h, 1, 0.7))
    color_palette.append(colorsys.hsv_to_rgb(h, 0.5, 0.7))
  color_palette = t.tensor(color_palette, dtype=t.float32)
  g = t.Generator()
  g.manual_seed(1)
  color_palette = color_palette[t.randperm(color_palette.shape[0], generator=g)]
  return color_palette
