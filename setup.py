from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'gpfanova',

  # ext_modules = cythonize(["gpfanova/*.pyx","gpfanova/sample/*.pyx","gpfanova/kernel/*.pyx"]),
  ext_modules = cythonize(["gpfanova/sample/*.pyx","gpfanova/kernel/*.pyx"]),

  version='0.1',
  description='Functional ANOVA using Gaussian Process priors.',
  author='Peter Tonner',
  email='peter.tonner@duke.edu',
  packages=['gpfanova','examples'],
  url='https://github.com/ptonner/gpfanova',

  classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Developers',
		  'Intended Audience :: Science/Research',
		  'Topic :: Scientific/Engineering',
          'Programming Language :: Python',
          ],

)
