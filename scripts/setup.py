#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

setup(
    scripts=["create_deformation.py"],
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("cython_code", ["cython_code.pyx"],
                   include_dirs = [numpy.get_include()])]
)
