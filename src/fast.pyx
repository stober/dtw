import numpy as np
import numpy.linalg as la
cimport numpy as np
DTYPE = np.double
ctypedef np.double_t DTYPE_t

cdef double min3(double a, double b, double c):
    m = a
    if b < m:
        m = b
    if c < m:
        m = c
    return m

cdef double dist(np.ndarray[DTYPE_t,ndim=1] x, np.ndarray[DTYPE_t,ndim=1] y):

    cdef int length = len(x)
    cdef unsigned int i
    cdef DTYPE_t d = 0

    for i in range(length):
        d += (x[i] - y[i]) ** 2

    return np.sqrt(d)
                                                        

def dtw_fast(np.ndarray s, np.ndarray t):
    
    cdef int nrows = s.shape[0]
    cdef int ncols = t.shape[0]
    
    cdef np.ndarray[DTYPE_t,ndim=2] dtw = np.zeros((nrows+1,ncols+1), dtype = DTYPE)

    dtw[:,0] = 1e6
    dtw[0,:] = 1e6
    dtw[0,0] = 0.0
    
    cdef unsigned int i,j
    cdef DTYPE_t cost
    
    for i in range(nrows):
        for j in range(ncols):
            cost = la.norm(s[i] - t[j])
            dtw[i+1,j+1] = cost + min3(dtw[i,j+1],dtw[i+1,j],dtw[i,j])

    return dtw[nrows,ncols]


def dtw_fast_2d(np.ndarray[DTYPE_t,ndim=2] s, np.ndarray[DTYPE_t,ndim=2] t):
    
    cdef int nrows = s.shape[0]
    cdef int ncols = t.shape[0]
    
    cdef np.ndarray[DTYPE_t,ndim=2] dtw = np.zeros((nrows+1,ncols+1), dtype = DTYPE)

    dtw[:,0] = 1e6
    dtw[0,:] = 1e6
    dtw[0,0] = 0.0
    
    cdef unsigned int i,j
    cdef DTYPE_t cost
    
    for i in range(nrows):
        for j in range(ncols):
            cost = dist(s[i],t[j])
            dtw[i+1,j+1] = cost + min3(dtw[i,j+1],dtw[i+1,j],dtw[i,j])

    return dtw[nrows,ncols]


# very fast
def dtw_fast_1d(np.ndarray[DTYPE_t,ndim=1] s, np.ndarray[DTYPE_t,ndim=1] t):
    
    cdef int nrows = s.shape[0]
    cdef int ncols = t.shape[0]
    
    cdef np.ndarray[DTYPE_t,ndim=2] dtw = np.zeros((nrows+1,ncols+1), dtype = DTYPE)

    dtw[:,0] = 1e6
    dtw[0,:] = 1e6
    dtw[0,0] = 0.0
    
    cdef unsigned int i,j
    cdef DTYPE_t cost
    
    for i in range(nrows):
        for j in range(ncols):
            cost = abs(s[i] - t[j])
            dtw[i+1,j+1] = cost + min3(dtw[i,j+1],dtw[i+1,j],dtw[i,j])

    return dtw[nrows,ncols]
