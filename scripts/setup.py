#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

setup(name="generate-deformation-fields",
    scripts=["create_deformation.py"],
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("cython_code_generate_deformation_fields", ["cython_code_generate_deformation_fields.pyx"],
                   include_dirs = [numpy.get_include()])]
)
