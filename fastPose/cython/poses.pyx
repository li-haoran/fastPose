#-----------------------
# fast overlaping calculate
#-----------------------

cimport cython
import numpy as np
cimport numpy as np

DTYPE = np.float
ctypedef np.float_t DTYPE_t

def pose_overlaps_cython(np.ndarray[DTYPE_t, ndim=2] poses,np.ndarray[DTYPE_t, ndim=2] query_poses):
	'''
	return overlaps
	'''
	cdef unsigned int N=poses.shape[0]
	cdef unsigned int K=query_poses.shape[0]
	cdef 