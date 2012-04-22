import numpy as np
cimport numpy as cnp

cnp.import_array()

cimport cython
from cython.parallel cimport prange

@cython.boundscheck(False)
@cython.wraparound(False)
def dot(cnp.ndarray[double, ndim=2, mode='c'] A,
        cnp.ndarray[double, ndim=2, mode='c'] B,
        cnp.ndarray[double, ndim=2, mode='c'] out):

    cdef:
        size_t rows_A, cols_A, rows_B, cols_B
        size_t i, j, k
        double s

    rows_A = A.shape[0]
    cols_A = A.shape[1]

    rows_B = B.shape[0]
    cols_B = B.shape[1]

    # Take each row of A
    for i in range(rows_A):

        # And multiply by every column of B
        for j in range(cols_B):
            s = 0
            for k in range(cols_A):
                s = s + A[i, k] * B[k, j]

            out[i, j] = s


@cython.boundscheck(False)
@cython.wraparound(False)
def pdot(cnp.ndarray[double, ndim=2, mode='c'] A,
         cnp.ndarray[double, ndim=2, mode='c'] B,
         cnp.ndarray[double, ndim=2, mode='c'] out):

    cdef:
        size_t rows_A, cols_A, rows_B, cols_B
        size_t i, j, k
        double s

    rows_A = A.shape[0]
    cols_A = A.shape[1]

    rows_B = B.shape[0]
    cols_B = B.shape[1]

    with nogil:

        # Take each row of A
        for i in prange(rows_A):

            # And multiply by every column of B
            for j in range(cols_B):
                s = 0
                for k in range(cols_A):
                    s = s + A[i, k] * B[k, j]

                out[i, j] = s
