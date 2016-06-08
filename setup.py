from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'GP_FANOVA',
  # ext_modules = cythonize(["gp_fanova/*.pyx","gp_fanova/sample/*.pyx","gp_fanova/kernel/*.pyx"]),
  ext_modules = cythonize(["gp_fanova/sample/*.pyx","gp_fanova/kernel/*.pyx"]),
)
