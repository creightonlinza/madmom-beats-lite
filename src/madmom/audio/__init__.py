# encoding: utf-8
"""
Audio package.

This lite runtime intentionally avoids eager imports of optional audio modules.
Import required processors directly from submodules, e.g.:

- ``madmom.audio.signal``
- ``madmom.audio.stft``
- ``madmom.audio.spectrogram``
"""

from __future__ import absolute_import, division, print_function

