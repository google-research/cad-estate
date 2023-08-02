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
"""Library for handling command line flags in a structured way."""

import argparse
import dataclasses
import enum
import re
import typing
from typing import Any, Sequence, Type, TypeVar

T = TypeVar('T')


class ArgType(enum.Enum):
  """The possible argument types"""
  FLAG = 1  # Flag, prefixed by a "--"
  POSITIONAL = 2  # Positional argument
  REMAINDER = 3  # The remaining arguments


FLAG = ArgType.FLAG
POSITIONAL = ArgType.POSITIONAL
REMAINDER = ArgType.REMAINDER


def flag(help_message: str, *, default: Any = None,
         arg_type: ArgType = ArgType.FLAG, short_name: str | None = None):
  """Marks a dataclass field as a command line flag.

  Args:
    help_message: The help message.
    default: The default value. If `None`, the flag becomes required. Has no
      effect on multi-value flags (ie fields with list type).
    arg_type: Whether this is a positional argument
    short_name: An optional alternative short name for the flag. "-" will be
      automatically prepended to this name.

  Returns:
    A dataclass field, populated with metadata corresponding to the arguments
    of this function.

  The flag name will match the field name, with "--" prepended to it.
  Supported field/flag types: `str`, `int`, `float`, `bool`, `List[str]`,
  `List[int]`, `List[float]`.
  """
  return dataclasses.field(
      default=default, metadata={
          "help": help_message,
          "arg_type": arg_type,
          "short_name": short_name
      })


def parse_flags(flag_struct_type: Type[T],
                flags: Sequence[str] | None = None) -> T:
  """Parses command line flags into a structured dataclass representation.

  Args:
    flag_struct_type: The class of the flags dataclass structure. The docstring
      of this class becomes the program description. Each field must be marked
      as a flag, using the `flag` function above.
    flags: The flags passed to the program. Will be taken from `sys.argv` by
      default.

  Returns:
    The parsed flags, filled into a new object of type `arg_type`.
  """
  list_flag_marker = object()
  list_default_args = {}
  parser = argparse.ArgumentParser(description=flag_struct_type.__doc__)
  for field in dataclasses.fields(flag_struct_type):
    field_meta = field.metadata
    help_message = field_meta["help"]
    short_name = field_meta["short_name"]
    arg_type = field_meta["arg_type"]

    if arg_type in {ArgType.POSITIONAL, ArgType.REMAINDER}:
      flag_name = [field.name]
    else:
      flag_name = ["--" + field.name]
      if short_name:
        flag_name += ["-" + short_name]

    default_value = field.default
    is_required = field.default is None

    flag_type = field.type
    is_list = typing.get_origin(field.type) == list
    if is_list:
      flag_type, = typing.get_args(flag_type)
      list_default_args[field.name] = default_value or []
      default_value = list_flag_marker
      is_required = False

    if flag_type in {str, int, float}:
      if arg_type == ArgType.POSITIONAL:
        kwargs = dict(nargs=("*" if is_list else None))
      elif arg_type == ArgType.REMAINDER:
        kwargs = dict(nargs="...")
      else:
        kwargs = dict(required=is_required, nargs=("*" if is_list else None))
      parser.add_argument(*flag_name, type=flag_type, default=default_value,
                          help=help_message, **kwargs)
    elif field.type == bool:
      assert not is_list
      group = parser.add_mutually_exclusive_group(required=is_required)
      group.add_argument(*flag_name, default=default_value, dest=field.name,
                         action="store_true", help=help_message)
      neg_flag_name = [re.sub(r"^(--?)", r"\1no", v) for v in flag_name]
      group.add_argument(*neg_flag_name, default=default_value, dest=field.name,
                         action="store_false", help=help_message)
    else:
      raise ValueError(
          f"Unsupported type '{field.type}' for argument '{field.name}'")
  result_args = parser.parse_args(args=flags)
  result_args = {
      k: (v if v != list_flag_marker else list_default_args[k])
      for k, v in vars(result_args).items()
  }
  return flag_struct_type(**result_args)
