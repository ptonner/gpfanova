from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'gpfanova',

  # ext_modules = cythonize(["gpfanova/*.pyx","gpfanova/sample/*.pyx","gpfanova/kernel/*.pyx"]),
  ext_modules = cythonize(["gpfanova/sample/*.pyx","gpfanova/kernel/*.pyx"]),

  version='0.1',
  description='Functional ANOVA using Gaussian Process priors.',
  author='Peter Tonner',
  author_email='peter.tonner@duke.edu',
  packages=['gpfanova','gpfanova.plot','gpfanova.sample','gpfanova.kernel','examples'],
  url='https://github.com/ptonner/gpfanova',

  keywords='bayesian statistics time-course',

  install_requires=[
	  	'scipy>=0.17.1',
		'numpy>=1.11.0',
		'pandas>=0.18.1',
		'Cython>=0.24',
		'matplotlib>=1.5.1',
	  ],

  classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Developers',
		  'Intended Audience :: Science/Research',
		  'Topic :: Scientific/Engineering',
          'Programming Language :: Python',
          ],

)
