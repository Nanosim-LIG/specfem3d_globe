#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ(2.61)
AC_INIT([seis_process], [1.0.0], [cig-seismo@geodynamics.org])
AC_CONFIG_SRCDIR([convolve_stf.c])
AC_CONFIG_HEADER([config.h])
AM_INIT_AUTOMAKE([foreign])

AC_ARG_VAR(SACAUX, [SAC 'aux' directory])
if test "x$SACAUX" = x; then
    AC_MSG_ERROR([environment variable SACAUX is not set])
fi
if test ! -d "$SACAUX"; then
    AC_MSG_ERROR([SACAUX (=$SACAUX) does not name a directory])
fi

# Checks for programs.
AC_PROG_CC

save_LDFLAGS=$LDFLAGS
save_CPPFLAGS=$CPPFLAGS
LDFLAGS="-L$SACAUX/../lib $LDFLAGS"
CPPFLAGS="-I$SACAUX/../include $CPPFLAGS"

# Checks for libraries.
AC_SEARCH_LIBS([sqrt], [m])
AC_CHECK_LIB([sacio], [rsac1], [], [AC_MSG_ERROR([sacio library not found])])
AC_CHECK_LIB([sac], [fft], [], [AC_MSG_ERROR([sac library not found])])

# Checks for header files.
AC_HEADER_STDC
AC_CHECK_HEADERS([limits.h stdlib.h string.h])
AC_CHECK_HEADER([sacio.h], [], [AC_MSG_ERROR([sacio.h not found])])
AC_CHECK_HEADER([sac.h], [], [AC_MSG_ERROR([sac.h not found])])

# Checks for typedefs, structures, and compiler characteristics.
AC_C_CONST

# Checks for library functions.
AC_FUNC_MALLOC
AC_FUNC_STRTOD
AC_CHECK_FUNCS([sqrt strtol])

# Check whether SAC uses 'int' instead of 'long' in function prototypes.
# (the current v101.3 -- and later? -- uses 'int').
AC_MSG_CHECKING([whether SAC prototypes use 'int' instead of 'long'])
AC_LANG_WERROR
AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
#include "sacio.h"
]], [[
    int nerr;
    getfhv(0,0,&nerr,0);
]])], [
    AC_MSG_RESULT(yes)
    AC_DEFINE([sac_int_t], [int], [SAC integer type.])
], [
    AC_MSG_RESULT(no)
    AC_DEFINE([sac_int_t], [long], [SAC integer type.])
])

LDFLAGS=$save_LDFLAGS
CPPFLAGS=$save_CPPFLAGS

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
