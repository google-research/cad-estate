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
"""File library with support for local and GCS file systems."""

import asyncio
import contextlib
import fnmatch
import glob
import logging
import os
import re
import typing as t

import aiofiles
import aiohttp
import aiohttp.client_exceptions
import backoff
import gcloud.aio.storage as aio_storage
import google.api_core.exceptions
from google.cloud import storage

_gcs_client: storage.Client | None = None
_gcs_async_client: aio_storage.Storage | None = None
log = logging.getLogger(__name__)
T = t.TypeVar("T")
NUM_GCS_RETRIES = 3

RECOVERABLE_ERRORS = (aiohttp.ClientResponseError,
                      aiohttp.client_exceptions.ClientError,
                      aiohttp.client_exceptions.ClientResponseError,
                      asyncio.TimeoutError)


def _should_giveup(e: Exception):
  if isinstance(e, aiohttp.ClientResponseError) and e.status == 404:
    return True
  return False


backoff_decorator = backoff.on_exception(
    backoff.expo, RECOVERABLE_ERRORS, max_tries=NUM_GCS_RETRIES,
    jitter=backoff.full_jitter, backoff_log_level=logging.DEBUG,
    giveup_log_level=logging.DEBUG, giveup=_should_giveup)


@contextlib.contextmanager
def auto_close_async_session():
  try:
    yield None
  finally:
    if _gcs_async_client:
      asyncio.get_event_loop().run_until_complete(_gcs_async_client.close())
      _gcs_async_client = None


def is_gs_path(p: str):
  return p.startswith("gs://")


def splitall(path: str):
  """Splits a path into all of its components."""
  result = []
  if is_gs_path(path):
    result.append(path[:5])
    path = path[5:]
  while True:
    head, tail = os.path.split(path)
    if head == path:
      result.append(head)
      break
    if tail == path:
      result.append(tail)
      break
    else:
      path = head
      result.append(tail)
  result.reverse()
  return result


def parse_gs_path(p: str):
  assert p.startswith("gs://")
  p = p[5:]
  parts = splitall(p)
  bucket, path = parts[0], "/".join(parts[1:])
  return bucket, path


def get_gcs_client():
  global _gcs_client
  if not _gcs_client:
    _gcs_client = storage.Client()

  return _gcs_client


def get_gcs_async_client():
  global _gcs_async_client
  if not _gcs_async_client:
    _gcs_async_client = aio_storage.Storage()

  return _gcs_async_client


def repeat_if_error(fn: t.Callable[[], T], num_tries, not_found_ok=False) -> T:
  for try_index in range(num_tries - 1):
    try:
      return fn()
    except (KeyboardInterrupt, SystemExit):
      raise
    except Exception as e:
      if isinstance(e, google.api_core.exceptions.NotFound) and not_found_ok:
        return None
      log.exception(f"Error in file operation, try={try_index}. Retrying ...")
  return fn()


def read_bytes(path: str) -> bytes:
  if is_gs_path(path):
    bucket_name, gcs_path = parse_gs_path(path)

    def _impl():
      bucket = get_gcs_client().get_bucket(bucket_name)
      return bucket.blob(gcs_path).download_as_string()

    return repeat_if_error(_impl, NUM_GCS_RETRIES)
  else:
    with open(path, "rb") as fl:
      return fl.read()


@backoff_decorator
async def read_bytes_async(path: str) -> bytes:
  if is_gs_path(path):
    bucket_name, gcs_path = parse_gs_path(path)
    client = get_gcs_async_client()
    return await client.download(bucket_name, gcs_path)
  else:
    async with aiofiles.open(path, "rb") as fl:
      return await fl.read()


async def await_in_parallel(awaitables: t.Collection[t.Awaitable[T]],
                            max_parallelism=50) -> list[T]:
  """Awaits tasks in parallel, with an upper bound on parallelism."""

  results = [None] * len(awaitables)

  async def await_result(index: int, awaitable: t.Awaitable[T]):
    results[index] = await awaitable

  remaining_tasks = [await_result(i, v) for i, v in enumerate(awaitables)]
  current_tasks = []
  while True:
    if current_tasks:
      done, pending = await asyncio.wait(current_tasks, timeout=5)
    else:
      done, pending = [], []
    for v in done:
      if e := v.exception():
        raise e
    current_tasks = list(pending)
    new_tasks = remaining_tasks[:max_parallelism]
    remaining_tasks = remaining_tasks[len(new_tasks):]
    current_tasks += [asyncio.create_task(v) for v in new_tasks]
    if not current_tasks:
      break

  return results


async def read_all_bytes_async(
    file_paths: t.Sequence[str], max_parallel_read_tasks: int = 50,
    progress_callback: t.Callable[[], None] | None = None) -> list[bytes]:
  """Reads binary files in parallel, using the async interface."""

  async def read_file(path: str):
    result = await read_bytes_async(path)
    if progress_callback:
      progress_callback()
    return result

  tasks = [read_file(v) for v in file_paths]
  return await await_in_parallel(tasks, max_parallel_read_tasks)


def read_text(path: str) -> str:
  return read_bytes(path).decode()


async def read_text_async(path: str) -> str:
  return (await read_bytes_async(path)).decode()


def write_bytes(path: str, contents: bytes):
  if is_gs_path(path):
    bucket_name, gcs_path = parse_gs_path(path)

    def _impl():
      bucket = get_gcs_client().get_bucket(bucket_name)
      bucket.blob(gcs_path).upload_from_string(contents)

    repeat_if_error(_impl, NUM_GCS_RETRIES)
  else:
    with open(path, "wb") as fl:
      fl.write(contents)


@backoff_decorator
async def write_bytes_async(path: str, contents: bytes):
  if is_gs_path(path):
    bucket_name, gcs_path = parse_gs_path(path)
    client = get_gcs_async_client()
    await client.upload(bucket_name, gcs_path, contents)
  else:
    async with aiofiles.open(path, "wb") as fl:
      await fl.write(contents)


async def write_all_bytes_async(  #
    paths_and_bytes: t.Iterable[t.Tuple[str, bytes]]):
  await asyncio.gather(*[write_bytes_async(p, b) for p, b in paths_and_bytes])


def write_all_bytes(paths_and_bytes: t.Iterable[t.Tuple[str, bytes]]):
  for k, v in paths_and_bytes:
    write_bytes(k, v)


def write_text(path: str, text: str):
  write_bytes(path, text.encode())


async def write_text_async(path: str, text: str):
  await write_bytes_async(path, text.encode())


def glob_pattern(pattern: str) -> t.Iterable[str]:
  if is_gs_path(pattern):
    bucket_name, gcs_path = parse_gs_path(pattern)

    parts = splitall(gcs_path)
    prefix = ""
    for part in parts:
      if re.match(r".*[?*\[].*", part):
        break
      prefix = os.path.join(prefix, part)

    def _impl():
      blobs = get_gcs_client().list_blobs(bucket_name, prefix=prefix)
      result = [
          f"gs://{bucket_name}/{v.name}" for v in blobs
          if fnmatch.fnmatch(v.name, gcs_path)
      ]
      return result

    return repeat_if_error(_impl, NUM_GCS_RETRIES)
  else:
    return glob.glob(pattern)


def unlink_file(path: str):
  if is_gs_path(path):
    bucket_name, gcs_path = parse_gs_path(path)
    return repeat_if_error(
        lambda: get_gcs_client().bucket(bucket_name).blob(gcs_path).delete(),
        NUM_GCS_RETRIES, not_found_ok=True)
  else:
    os.unlink(path)


def rename_file(old_path: str, new_path: str):
  if is_gs_path(old_path) != is_gs_path(new_path):
    log.error("Invalid rename (different file systems): "
              f"'{old_path}'->'{new_path}'")
    raise ValueError("Both files must be on the same file system")
  if is_gs_path(old_path):
    bucket_name, old_gcs_path = parse_gs_path(old_path)
    _, new_gcs_path = parse_gs_path(new_path)

    def _impl():
      bucket = get_gcs_client().bucket(bucket_name)
      bucket.rename_blob(bucket.blob(old_gcs_path), new_gcs_path)

    return repeat_if_error(_impl, NUM_GCS_RETRIES)
  else:
    os.rename(old_path, new_path)


def make_dirs(path: str):
  if is_gs_path(path):
    return
  os.makedirs(path, exist_ok=True)


def join(*args):
  if len(args) == 1:
    return args[0]
  for i, v in enumerate(args[1:]):
    if is_gs_path(v):
      return join(*args[i + 1:])
  return os.path.join(*args)


def normpath(p: str):
  if is_gs_path(p):
    return f"gs://{os.path.normpath(p[5:])}"
  return os.path.normpath(p)


def isabs(p: str):
  if is_gs_path(p):
    return True
  return os.path.isabs(p)


def dirname(p: str):
  return os.path.dirname(p)


def abspath(p: str):
  if is_gs_path(p):
    return p
  return os.path.abspath(os.path.expanduser(p))


def basename(p: str):
  return os.path.basename(p)


def relpath(p: str, prefix: str) -> str:
  if is_gs_path(p) != is_gs_path(prefix):
    raise ValueError("Both paths have to be on the same storage system "
                     "(either GCS or local)")
  if is_gs_path(p):
    p = p[5:]
    prefix = prefix[5:]
  return os.path.relpath(p, prefix)


def splitext(p: str):
  return os.path.splitext(p)


@backoff_decorator
def exists(path: str):
  if is_gs_path(path):
    bucket_name, gcs_path = parse_gs_path(path)
    bucket = get_gcs_client().bucket(bucket_name)
    return bucket.blob(gcs_path).exists()
  else:
    return os.path.exists(path)
