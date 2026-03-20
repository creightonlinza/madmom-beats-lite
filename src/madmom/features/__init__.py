# encoding: utf-8
"""
High-level feature package.

This lite runtime intentionally avoids importing all feature modules at package
import time. Import required processors directly from submodules, e.g.:

- ``madmom.features.downbeats``
- ``madmom.features.beats``
"""

from __future__ import absolute_import, division, print_function

