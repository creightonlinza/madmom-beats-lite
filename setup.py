from __future__ import annotations

from distutils.extension import Extension
import numpy as np
from Cython.Build import cythonize
from setuptools import setup

extensions = [
    Extension(
        "madmom.ml.hmm",
        ["src/madmom/ml/hmm.pyx"],
        include_dirs=[np.get_include()],
    ),
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
    )
)
