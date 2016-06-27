from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'gpfanova',
  # ext_modules = cythonize(["gpfanova/*.pyx","gpfanova/sample/*.pyx","gpfanova/kernel/*.pyx"]),
  ext_modules = cythonize(["gpfanova/sample/*.pyx","gpfanova/kernel/*.pyx"]),
)
