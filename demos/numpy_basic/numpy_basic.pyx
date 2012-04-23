cimport numpy as cnp


def foo(cnp.ndarray[cnp.float64_t, ndim=2] arr):
    cdef size_t i, j
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            arr[i, j] = i + j

