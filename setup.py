# from distutils.core import setup
from setuptools import setup, find_packages, Extension

use_cython=True
try:
    from Cython.Distutils import build_ext
    from Cython.Build import cythonize
except ImportError:
    use_cython = False

cmdclass = { }
ext_modules = [ ]

if use_cython:
    ext_modules += cythonize(["gpfanova/sample/*.pyx","gpfanova/kernel/*.pyx"])
    # ext_modules += [
    #     Extension("gpfanova.sample", ["gpfanova/sample/*.pyx"]),
	# 	Extension("gpfanova.kernel", ["gpfanova/kernel/*.pyx"]),
    # ]
    cmdclass.update({ 'build_ext': build_ext })
else:
    ext_modules += [
        # Extension("gpfanova.sample", ["gpfanova/sample/*.c"]),
		# Extension("gpfanova.kernel", ["gpfanova/kernel/*.c"]),
		Extension("gpfanova.kernel.kernel", ["gpfanova/kernel/kernel.c"]),
		Extension("gpfanova.kernel.rbf", ["gpfanova/kernel/rbf.c"]),
		Extension("gpfanova.sample.slice", ["gpfanova/sample/slice.c"]),
    ]

setup(
  name = 'gpfanova',

  # ext_modules = cythonize(["gpfanova/*.pyx","gpfanova/sample/*.pyx","gpfanova/kernel/*.pyx"]),
  # ext_modules = cythonize(["gpfanova/sample/*.pyx","gpfanova/kernel/*.pyx"]),
  cmdclass = cmdclass,
  ext_modules=ext_modules,

  version='0.1.14',
  description='Functional ANOVA using Gaussian Process priors.',
  author='Peter Tonner',
  author_email='peter.tonner@duke.edu',
  # packages=['gpfanova','gpfanova.plot','gpfanova.sample','gpfanova.kernel','examples'],
  packages = find_packages(exclude=('analysis*','analysis.*','data','results')),
  url='https://github.com/ptonner/gpfanova',

  keywords='bayesian statistics time-course',

  install_requires=[
	  	'scipy>=0.17.1',
		'numpy>=1.11.0',
		'pandas>=0.18.1',
		# 'Cython>=0.24',
		'matplotlib>=1.5.1',
		'patsy>=0.4.1'
	  ],

  classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Developers',
		  'Intended Audience :: Science/Research',
		  'Topic :: Scientific/Engineering',
          'Programming Language :: Python',
          ],

)
