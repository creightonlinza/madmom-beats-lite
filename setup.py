from __future__ import annotations

import sys
from distutils.extension import Extension
import numpy as np
from Cython.Build import cythonize
from setuptools import setup

# Contracted multiply-adds round differently, which would change the LSTM output.
strict_fp_args = [] if sys.platform == "win32" else ["-ffp-contract=off"]

extensions = [
    Extension(
        "madmom.ml.hmm",
        ["src/madmom/ml/hmm.pyx"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        "madmom.ml.nn._lstm",
        ["src/madmom/ml/nn/_lstm.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=strict_fp_args,
    ),
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
    )
)
