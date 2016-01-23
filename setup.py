# This file is part of RankPy.
# 
# RankPy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Lesser Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RankPy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Lesser Public License for more details.
#
# You should have received a copy of the GNU General Lesser Public License
# along with Foobar.  If not, see <http://www.gnu.org/licenses/>.

import os
import numpy

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

if __name__ == '__main__':
    rankpy_dir = os.path.join(os.path.dirname(__file__), 'rankpy')

    libraries = []
    if os.name == 'posix':
        libraries.append('m')

    setup(
        name = 'RankPy',
        version = '0.0.1-alpha',
        author = 'Tomas Tunys',
        author_email = 'tunystom@gmail.com',
        description = ('This project is designed to make freely available fast implementations of the current state-of-the-art methods for learning to rank in Python.'),
        long_description=open('README.rst', 'r').read(),
        keywords = 'machine learning, learning to rank, information retrieval',
        url = 'https://bitbucket.org/tunystom/rankpy',
        download_url = 'https://bitbucket.org/tunystom/rankpy/downloads',
        ext_modules=[Extension('rankpy.metrics._metrics',
                               sources=['./rankpy/metrics/_metrics.c'],
                               include_dirs=[numpy.get_include()],
                               libraries=libraries),
                     Extension('rankpy.metrics._utils',
                               sources=['./rankpy/metrics/_utils.c'],
                               include_dirs=[numpy.get_include()],
                               libraries=libraries),
                     Extension('rankpy.models.lambdamart_inner',
                               sources=['./rankpy/models/lambdamart_inner.c'],
                               include_dirs=[numpy.get_include()],
                               libraries=libraries),
                     Extension('rankpy.tree._tree',
                               sources=['./rankpy/tree/_tree.c'],
                               include_dirs=[numpy.get_include()],
                               libraries=libraries,
                               extra_compile_args=['-O3'])],
        cmdclass={'build_ext': build_ext},
        packages=find_packages(),
        license = 'GNU General Lesser Public License v3 or later (GPLv3+)',
        classifiers=['Development Status :: 3 - Alpha',
                     'Environment :: Console',
                     'Intended Audience :: Science/Research',
                     'License :: OSI Approved :: GNU General Lesser Public License v3 or later (GPLv3+)',
                     'Operating System :: OS Independent',
                     'Programming Language :: Python :: 2.7',
                     'Topic :: Scientific/Engineering :: Artificial Intelligence']
    )
