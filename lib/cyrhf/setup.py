# -*- coding: utf-8 -*-
# @Author: andrian
# @Date:   2020-12-08 15:55:01
# @Last Modified by:   andrian
# @Last Modified time: 2021-02-08 12:53:58

from setuptools import Extension, setup
from Cython.Build import cythonize

import numpy

# setup(
#     # setup_requires=[
#     #     # Setuptools 18.0 properly handles Cython extensions.
#     #     'setuptools>=18.0',
#     #     'cython',
#     # ],
#     ext_modules = [Extension("rhf", ["rhf.c"], include_dirs=[numpy.get_include()])],
#     name='rhf',
#     version = '0.2.0',
# )

setup(
    name='rhf',
    ext_modules=cythonize('rhf.pyx'),
    include_dirs=[numpy.get_include()]
)