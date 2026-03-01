from __future__ import annotations

from setuptools import Extension, setup

import numpy as np
from Cython.Build import cythonize

extensions = [
    Extension(
        "madmom_beats_lite._vendor.madmom_lite.ml.hmm",
        ["src/madmom_beats_lite/_vendor/madmom_lite/ml/hmm.pyx"],
        include_dirs=[np.get_include()],
    )
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": 3, "boundscheck": False, "wraparound": False},
    )
)
