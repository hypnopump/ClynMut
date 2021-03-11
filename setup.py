from setuptools import setup, find_packages

setup(
  name = 'alphafold2-pytorch',
  packages = find_packages(),
  version = '0.0.50',
  license='MIT',
  description = 'ClynMut',
  author = 'Eric Alcaide',
  author_email = 'ericalcaide1@gmail.com',
  url = 'https://github.com/hypnopump/clynmut',
  keywords = [
    'artificial intelligence',
    'bioinformatics',
    'mutation prediction' 
  ],
  install_requires=[
    'einops>=0.3',
    'numpy',
    'torch>=1.6',
    'mdtraj>=1.8',
    'tqdm',
    'alphafold2-pytorch'
  ],
  setup_requires=[
    'pytest-runner',
  ],
  tests_require=[
    'pytest'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)