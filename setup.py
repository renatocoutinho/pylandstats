# coding=utf-8

from io import open  # compatible enconding parameter
from os import path

from pythran.dist import PythranBuildExt, PythranExtension
from setuptools import dist, find_packages, setup

__version__ = '0.5.0'

classifiers = [
    'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
    'Programming Language :: Python',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
]

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# get the dependencies and installs
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    all_reqs = f.read().split('\n')

# Extra dependencies for geometric operations
# we deliberately do not set any lower nor upper bounds on `geopandas`
# dependency so that people might install its cythonized version
geo = ["geopandas", "shapely >= 1.0.0"]

install_requires = [x.strip() for x in all_reqs if 'git+' not in x]
dependency_links = [
    x.strip().replace('git+', '') for x in all_reqs if x.startswith('git+')
]

# This is required to be able to use pythran in setup.py
dist.Distribution(dict(setup_requires='pythran'))

setup(
    name='pylandstats',
    version=__version__,
    description='Open-source Python library to compute landscape metrics',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=classifiers,
    url='https://github.com/martibosch/pylandstats',
    author='Martí Bosch',
    author_email='marti.bosch@epfl.ch',
    license='GPL-3.0',
    packages=find_packages(exclude=['docs', 'tests*']),
    include_package_data=True,
    install_requires=install_requires,
    extras_require={'geo': geo},
    dependency_links=dependency_links,
    ext_modules=[
        PythranExtension('pylandstats_compute', ['pylandstats/compute.py'])
    ],
    cmdclass={'build_ext': PythranBuildExt},
)
