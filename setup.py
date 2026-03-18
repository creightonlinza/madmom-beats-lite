from __future__ import annotations

from distutils.extension import Extension
from pathlib import Path

import numpy as np
from setuptools import setup

hmm_pyx = Path("src/madmom/ml/hmm.pyx")
hmm_c = Path("src/madmom/ml/hmm.c")
hmm_source = str(hmm_pyx if hmm_pyx.exists() else hmm_c)

extensions = [
    Extension(
        "madmom.ml.hmm",
        [hmm_source],
        include_dirs=[np.get_include()],
    ),
]

if hmm_source.endswith(".pyx"):
    from Cython.Build import cythonize

    ext_modules = cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
    )
else:
    ext_modules = extensions

setup(ext_modules=ext_modules)
