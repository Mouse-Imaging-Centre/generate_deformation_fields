#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.
# Require autoconf 2.61 or newer.
AC_PREREQ(2.61)
AC_INIT([assess-registration],[0.1],[Matthijs van Eede matthijs@phenogenomics.ca])
AC_CONFIG_SRCDIR([src/create_determinant_file.cpp])

AM_INIT_AUTOMAKE

# The argument is the name of the generated header file.
# It is recommended to leave it as "config.h".
AC_CONFIG_HEADERS([config.h])

# This macro eases building the package; see m4/README.
smr_WITH_BUILD_PATH

# Checks for programs.
AC_PROG_CXX
AC_PROG_CC
AC_PROG_INSTALL

# Checks for libraries.  See m4/README.
mni_REQUIRE_BICPL

# Checks for header files.
AC_HEADER_STDC
AC_CHECK_HEADERS([stdlib.h string.h unistd.h])

# Checks for library functions.
AC_FUNC_MALLOC
AC_CHECK_FUNCS([strtol])

# Finish up by writing output files.
AC_CONFIG_FILES([Makefile
                 scripts/Makefile
                 src/Makefile])
AC_OUTPUT

