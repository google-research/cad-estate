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
"""Routines for working with OpenGL EGL contexts."""

import ctypes
import logging
import os
from typing import Dict, Optional, Text

import glcontext

log = logging.getLogger(__name__)

# Keeps the OpenGL context active for the specified CUDA device (if >= 0).
keep_context_active_for_cuda_device = -1


class EGL:
  EGLAttrib = ctypes.c_ssize_t
  EGLBoolean = ctypes.c_bool
  EGLConfig = ctypes.c_void_p
  EGLContext = ctypes.c_void_p
  EGLDeviceEXT = ctypes.c_void_p
  EGLDisplay = ctypes.c_void_p
  EGLSurface = ctypes.c_void_p
  EGLenum = ctypes.c_uint
  EGLint = ctypes.c_int32

  EGL_BLUE_SIZE = 0x3022
  EGL_CUDA_DEVICE_NV = 0x323A
  EGL_DEPTH_SIZE = 0x3025
  EGL_DRM_DEVICE_FILE_EXT = 0x3233
  EGL_EXTENSIONS = 0x3055
  EGL_GREEN_SIZE = 0x3023
  EGL_NONE = 0x3038
  EGL_NO_CONTEXT = 0
  EGL_NO_SURFACE = 0
  EGL_OPENGL_API = 0x30A2
  EGL_OPENGL_BIT = 0x0008
  EGL_PBUFFER_BIT = 0x0001
  EGL_PLATFORM_DEVICE_EXT = 0x313F
  EGL_RED_SIZE = 0x3024
  EGL_RENDERABLE_TYPE = 0x3040
  EGL_SUCCESS = 0x3000
  EGL_SURFACE_TYPE = 0x3033
  EGL_VENDOR = 0x3053

  def __init__(self):
    egl_lib_path = os.getenv("LIB_EGL_PATH", "libEGL.so.1")
    log.debug(f"Initializing EGL using library {egl_lib_path}")
    self.egl_lib = ctypes.cdll.LoadLibrary(egl_lib_path)

    self.eglGetProcAddress = self.egl_lib.eglGetProcAddress
    self.eglGetProcAddress.argtypes = [ctypes.c_char_p]
    self.eglGetProcAddress.restype = ctypes.c_void_p

    self.eglGetError = ctypes.CFUNCTYPE(EGL.EGLint)(
        self.load_function(b"eglGetError"))

    self.eglQueryDevicesEXT = ctypes.CFUNCTYPE(
        EGL.EGLBoolean, EGL.EGLint, ctypes.POINTER(EGL.EGLDeviceEXT),
        ctypes.POINTER(EGL.EGLint))(
            self.load_function(b"eglQueryDevicesEXT"))

    self.eglQueryDeviceAttribEXT = ctypes.CFUNCTYPE(
        EGL.EGLBoolean, EGL.EGLDeviceEXT, EGL.EGLint,
        ctypes.POINTER(EGL.EGLAttrib))(
            self.load_function(b"eglQueryDeviceAttribEXT"))

    self.eglQueryDeviceStringEXT = ctypes.CFUNCTYPE(
        ctypes.c_char_p, EGL.EGLDeviceEXT, EGL.EGLint)(
            self.load_function(b"eglQueryDeviceStringEXT"))

    self.eglGetPlatformDisplayEXT = ctypes.CFUNCTYPE(
        EGL.EGLDisplay, EGL.EGLenum, EGL.EGLDeviceEXT,
        ctypes.POINTER(EGL.EGLint))(
            self.load_function(b"eglGetPlatformDisplayEXT"))

    self.eglQueryString = ctypes.CFUNCTYPE(
        ctypes.c_char_p, EGL.EGLDisplay, EGL.EGLint)(
            self.load_function(b"eglQueryString"))

    self.eglInitialize = ctypes.CFUNCTYPE(
        EGL.EGLBoolean, EGL.EGLDisplay, ctypes.POINTER(EGL.EGLint),
        ctypes.POINTER(EGL.EGLint))(
            self.load_function(b"eglInitialize"))

    self.eglBindAPI = ctypes.CFUNCTYPE(EGL.EGLBoolean, EGL.EGLenum)(
        self.load_function(b"eglBindAPI"))

    self.eglChooseConfig = ctypes.CFUNCTYPE(
        EGL.EGLBoolean, EGL.EGLDisplay, ctypes.POINTER(EGL.EGLint),
        ctypes.POINTER(EGL.EGLConfig), EGL.EGLint, ctypes.POINTER(EGL.EGLint))(
            self.load_function(b"eglChooseConfig"))

    self.eglCreateContext = ctypes.CFUNCTYPE(
        EGL.EGLContext, EGL.EGLDisplay, EGL.EGLConfig, EGL.EGLContext,
        ctypes.POINTER(EGL.EGLint))(
            self.load_function(b"eglCreateContext"))

    self.eglMakeCurrent = ctypes.CFUNCTYPE(
        EGL.EGLBoolean, EGL.EGLDisplay, EGL.EGLSurface, EGL.EGLSurface,
        EGL.EGLContext)(
            self.load_function(b"eglMakeCurrent"))

    self.eglGetCurrentContext = ctypes.CFUNCTYPE(EGL.EGLContext)(
        self.load_function(b"eglGetCurrentContext"))

    self.eglDestroyContext = ctypes.CFUNCTYPE(
        EGL.EGLBoolean, EGL.EGLDisplay, EGL.EGLContext)(
            self.load_function(b"eglDestroyContext"))

  def load_function(self, name: bytes):
    result = self.eglGetProcAddress(name)
    if not result:
      raise ValueError(f"Failed to load function '{name.decode()}'")
    return result


_eglInstance: Optional[EGL] = None


def _egl():
  global _eglInstance
  if not _eglInstance:
    _eglInstance = EGL()
  return _eglInstance


class ContextError(Exception):
  pass


class EglContext:
  _context_cache: Dict[int, EGL.EGLDisplay] = {}
  _cuda_to_egl_mapping: Dict[int, EGL.EGLDeviceEXT] = {}

  def __init__(self, cuda_device: int):
    if cuda_device in EglContext._context_cache:
      log.debug(f"Using cached EGL context for cuda device {cuda_device}")
      self._display = EglContext._context_cache[cuda_device]
    else:
      log.debug(f"Creating EGL context for cuda device {cuda_device}")
      self._display = self._create_display(cuda_device)
      EglContext._context_cache[cuda_device] = self._display
    self._context = self._create_context(self._display)
    self._cuda_device = cuda_device
    self.__enter__()

  @classmethod
  def _egl_check_success(cls, pred: bool, msg: Text, level=logging.FATAL):
    err = _egl().eglGetError()
    if not pred or err != EGL.EGL_SUCCESS:
      msg = f"{msg}. EGL error is: 0x{err:x}."
      log.log(level, msg)
      if level >= logging.FATAL:
        raise ContextError(msg)
      return False
    return True

  @classmethod
  def _get_egl_device_map(cls):
    if not cls._cuda_to_egl_mapping:
      max_devices = 64
      devices = (EGL.EGLDeviceEXT * max_devices)()
      num_devices = EGL.EGLint()
      cls._egl_check_success(
          _egl().eglQueryDevicesEXT(max_devices, devices,
                                    ctypes.pointer(num_devices)),
          "Failed to retrieve EGL devices")
      log.debug(f"Found {num_devices} GPUs with EGL support.")

      for device_idx, device in enumerate(list(devices[:num_devices.value])):
        device_extensions = (_egl().eglQueryDeviceStringEXT(
            device, EGL.EGL_EXTENSIONS))  # type: bytes
        cls._egl_check_success(
            bool(device_extensions), "Unable to retrieve device extensions")
        device_extensions_set = set(device_extensions.split(b" "))
        if b"EGL_NV_device_cuda" not in device_extensions_set:
          log.debug(f"Ignoring EGL device {device_idx}. "
                    f"No support for EGL_NV_device_cuda.")
          continue

        cuda_device_attr = EGL.EGLAttrib(-1)
        st = _egl().eglQueryDeviceAttribEXT(device, EGL.EGL_CUDA_DEVICE_NV,
                                            ctypes.pointer(cuda_device_attr))
        if not st or _egl().eglGetError() != EGL.EGL_SUCCESS:
          log.debug(f"Unable to get CUDA device for EGL device {device_idx}")
          continue
        cuda_device = cuda_device_attr.value

        cls._cuda_to_egl_mapping[cuda_device] = device
    return cls._cuda_to_egl_mapping

  def _create_display(self, cuda_device: int):
    devices = self._get_egl_device_map()
    if cuda_device not in devices:
      raise ContextError(
          f"Could not find EGL device for CUDA device {cuda_device}")
    device = devices[cuda_device]
    display = _egl().eglGetPlatformDisplayEXT(EGL.EGL_PLATFORM_DEVICE_EXT,
                                              device, None)
    self._egl_check_success(bool(display), "Unable to create EGL display")
    major, minor = EGL.EGLint(), EGL.EGLint()
    self._egl_check_success(
        _egl().eglInitialize(display, ctypes.pointer(major),
                             ctypes.pointer(minor)),
        "Unable to initialize display")

    self._egl_check_success(_egl().eglBindAPI(EGL.EGL_OPENGL_API),
                            "Unable to bind OpenGL API to display")

    return display

  def _create_context(self, display: EGL.EGLDisplay):
    config_attrib_list = [
        EGL.EGL_SURFACE_TYPE, EGL.EGL_PBUFFER_BIT, EGL.EGL_RED_SIZE, 8,
        EGL.EGL_GREEN_SIZE, 8, EGL.EGL_BLUE_SIZE, 8, EGL.EGL_DEPTH_SIZE, 8,
        EGL.EGL_RENDERABLE_TYPE, EGL.EGL_OPENGL_BIT, EGL.EGL_NONE
    ]
    ct_config_attrib_list = (EGL.EGLint *
                             len(config_attrib_list))(*config_attrib_list)

    config = EGL.EGLConfig()
    num_config = EGL.EGLint()
    is_successful = _egl().eglChooseConfig(display, ct_config_attrib_list,
                                           ctypes.pointer(config), 1,
                                           ctypes.pointer(num_config))
    self._egl_check_success(is_successful and num_config.value == 1,
                            "Unable to choose EGL config")

    context_attrib = EGL.EGLint(EGL.EGL_NONE)
    context = _egl().eglCreateContext(display, config, None,
                                      ctypes.pointer(context_attrib))
    self._egl_check_success(context, "Unable to create context")
    return context

  def load(self, name: Text) -> int:
    log.debug(f"Loading function {name}")
    return _egl().load_function(name.encode())

  def _check_valid_context(self):
    if not self._context:
      raise ContextError("Context already released!")

  def __enter__(self):
    self._check_valid_context()

    if _egl().eglGetCurrentContext() == self._context:
      return

    self._egl_check_success(
        _egl().eglMakeCurrent(self._display, EGL.EGL_NO_SURFACE,
                              EGL.EGL_NO_SURFACE, self._context),
        "Unable to make context current")

  def __exit__(self, *args):
    if (_egl().eglGetCurrentContext() == self._context
        and self._cuda_device == keep_context_active_for_cuda_device):
      return

    self._egl_check_success(
        _egl().eglMakeCurrent(self._display, EGL.EGL_NO_SURFACE,
                              EGL.EGL_NO_SURFACE, EGL.EGL_NO_CONTEXT),
        "Unable to release context")

  def release(self):
    self._check_valid_context()

    if _egl().eglGetCurrentContext == self._context:
      self._egl_check_success(
          _egl().eglMakeCurrent(self._display, EGL.EGL_NO_SURFACE,
                                EGL.EGL_NO_SURFACE, EGL.EGL_NO_CONTEXT),
          "Unable to release context")

    self._egl_check_success(
        _egl().eglDestroyContext(self._display, self._context),
        "Unable to destroy context")
    self._context = None


def create_moderngl_context(*args, **kwargs):
  assert len(args) == 0
  if "cuda_device" not in kwargs:
    raise ContextError("cuda_device must be specified.")
  cuda_device = kwargs["cuda_device"]
  return EglContext(cuda_device)


def monkey_patch_moderngl():
  old_fn = glcontext.get_backend_by_name

  def get_backend_by_name(name):
    if name == "cuda_egl":
      return create_moderngl_context
    else:
      return old_fn(name)

  glcontext.get_backend_by_name = get_backend_by_name
