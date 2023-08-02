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
"""Various debugging helpers."""

import base64
import io
import logging
import re
import typing

import numpy as np
import PIL.Image
import torch as t
from IPython.core import display

log = logging.getLogger(__name__)


def to_hwc_rgb8(imgarr: typing.Any) -> np.ndarray:
  if t.is_tensor(imgarr):  # Torch -> Numpy
    imgarr = imgarr.detach().cpu().numpy()
  if hasattr(imgarr, "numpy"):  # TF -> Numpy
    imgarr = imgarr.numpy()
  if len(imgarr.shape) == 2:  # Monochrome -> RGB
    imgarr = np.stack([imgarr] * 3, -1)
  if (len(imgarr.shape) == 3 and imgarr.shape[0] <= 4
      and (imgarr.shape[1] > 4 or imgarr.shape[2] > 4)):  # CHW -> HWC
    imgarr = np.transpose(imgarr, [1, 2, 0])
  if len(imgarr.shape) == 3 and imgarr.shape[-1] == 4:  # RGBA -> RGB
    imgarr = imgarr[:, :, :3]
  if len(imgarr.shape) == 3 and imgarr.shape[-1] == 1:  # Monochrome -> RGB
    imgarr = np.concatenate([imgarr] * 3, -1)
  if imgarr.dtype == np.float32 or imgarr.dtype == np.float64:
    imgarr = np.minimum(np.maximum(imgarr * 255, 0), 255).astype(np.uint8)
  if imgarr.dtype == np.int32 or imgarr.dtype == np.int64:
    imgarr = np.minimum(np.maximum(imgarr, 0), 255).astype(np.uint8)
  if imgarr.dtype == np.bool_:
    imgarr = imgarr.astype(np.uint8) * 255

  if (len(imgarr.shape) != 3 or imgarr.shape[-1] != 3
      or imgarr.dtype != np.uint8):
    raise ValueError(
        "Cannot display image from array with type={} and shape={}".format(
            imgarr.dtype, imgarr.shape))

  return imgarr[..., :3]


def image_as_url(imgarr: np.ndarray, fmt: str = "png") -> str:
  img = PIL.Image.fromarray(imgarr, "RGB")
  buf = io.BytesIO()
  img.save(buf, fmt)
  b64 = base64.encodebytes(buf.getvalue()).decode("utf8")
  b64 = "data:image/png;base64,{}".format(b64)
  return b64


class Image(typing.NamedTuple):
  image: typing.Any
  label: str
  width: int


def get_html_for_images(*orig_images, fmt="png", w=None):
  table_template = """
    <div style="display: inline-flex; flex-direction: row; flex-wrap:wrap">
      {}
    </div>
  """
  item_template = """
    <div style="display: inline-flex; flex-direction: column; flex-wrap:
         nowrap; align-items: center">
      <img style="margin-right: 0.5em" src="{image}" width="{width}"/>
      <div style="margin-bottom: 0.5em; margin-right: 0.5em">{label}</div>
    </div>
  """
  images = []

  def append_image(image):
    image = to_hwc_rgb8(image)
    width = image.shape[1] if not w else w
    images.append(Image(label="Image {}".format(idx), image=image, width=width))

  for idx, item in enumerate(orig_images):
    if isinstance(item, str) and images:
      images[-1] = images[-1]._replace(label=item)
    elif isinstance(item, bytes):
      image = np.array(PIL.Image.open(io.BytesIO(item)))
      append_image(image)
    elif isinstance(item, PIL.Image.Image):
      append_image(np.array(item))
    elif isinstance(item, int) and images:
      images[-1] = images[-1]._replace(width=item)
    else:
      append_image(item)

  images = [v._replace(image=image_as_url(v.image, fmt)) for v in images]
  table = [item_template.format(**v._asdict()) for v in images]
  table = table_template.format("".join(table))
  return table


def display_images(*orig_images, **kwargs):
  """Display images in a IPython environment"""
  display.display(display.HTML(get_html_for_images(*orig_images, **kwargs)))


def print_tensor(v: t.Tensor):
  v = v.detach()
  dtype = re.sub(r"^torch\.", "", str(v.dtype))
  sep = "\n" if len(v.shape) > 1 else " "
  return f"{dtype}{list(v.shape)}({v.device}){{{sep}{v.cpu().numpy()}{sep}}}"


def better_tensor_display():
  """Better string representation of tensors for python debuggers."""
  np.set_printoptions(4, suppress=True)
  t.set_printoptions(4, sci_mode=False)
  t.Tensor.__repr__ = print_tensor


def better_jupyter_display():
  try:

    def _print_key_dict(v, p, cycle):
      p.text(str(list(v)))

    formatters = get_ipython().display_formatter.formatters['text/plain']
    formatters.for_type("collections.abc.KeysView", _print_key_dict)
  except Exception as e:
    log.exception("Unable to instrument Jupyter notebook")


def dump_state(path: str, **kwargs):
  state_dict = {}
  for k, v in kwargs.items():
    assert isinstance(k, str)
    if t.is_tensor(v):
      v = v.detach().cpu()
    state_dict[k] = v
  t.save(state_dict, path)
