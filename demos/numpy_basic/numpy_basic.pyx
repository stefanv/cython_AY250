cimport numpy as np


def foo(np.ndarray[np.float64_t, ndim=2] arr):
    cdef int i, j
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            arr[i, j] = i + j

