#!python
#cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
# _cy_math.pyx
# All math utilities are declared as inline cdef in _cy_math.pxd.
# This stub exists so Cython builds _cy_math as an extension module;
# the pxd body is inlined into every module that cimports from it.

# No standalone symbols needed here — everything is inline in the pxd.
