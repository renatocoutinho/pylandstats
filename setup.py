# coding=utf-8

from distutils.command.build_ext import build_ext
from distutils.sysconfig import customize_compiler
from io import open  # compatible enconding parameter
from os import path

from setuptools import find_packages, setup
from utils import cc

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


# Avoid a gcc warning (see https://stackoverflow.com/questions/8106258/cc1plus-
# warning-command-line-option-wstrict-prototypes-is-valid-for-ada-c-o):
# cc1plus: warning: command line option ‘-Wstrict-prototypes’ is valid
# for C/ObjC but not for C++
# See also: https://github.com/numba/numba/issues/3361
class BuildExt(build_ext):
    def build_extensions(self):
        customize_compiler(self.compiler)
        try:
            self.compiler.compiler_so.remove("-Wstrict-prototypes")
        except (AttributeError, ValueError):
            pass
        build_ext.build_extensions(self)


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
    cmdclass={'build_ext': BuildExt},
    ext_modules=[cc.distutils_extension()],
)
