# encoding: utf-8
"""Minimal utility symbols required by vendored madmom-lite components."""

from __future__ import absolute_import, division, print_function

import argparse
import io

import numpy as np

try:
    string_types = basestring
    integer_types = (int, long, np.integer)
except NameError:
    string_types = str
    integer_types = (int, np.integer)

try:
    file_types = (io.IOBase, file)
except NameError:
    file_types = io.IOBase


class OverrideDefaultListAction(argparse.Action):
    """argparse action that replaces default list values on first use."""

    def __init__(self, sep=None, *args, **kwargs):
        super(OverrideDefaultListAction, self).__init__(*args, **kwargs)
        self.set_to_default = True
        self.list_type = self.type
        if sep is not None:
            self.type = str
        self.sep = sep

    def __call__(self, parser, namespace, value, option_string=None):
        if self.set_to_default:
            setattr(namespace, self.dest, [])
            self.set_to_default = False
        cur_values = getattr(namespace, self.dest)
        try:
            cur_values.extend([self.list_type(v) for v in value.split(self.sep)])
        except ValueError as exc:
            raise argparse.ArgumentError(self, str(exc) + value)
