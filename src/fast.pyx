import numpy as np
import numpy.linalg as la
cimport numpy as np

DTYPE = np.double
ctypedef np.double_t DTYPE_t

ITYPE = np.int
ctypedef np.int_t ITYPE_t

cdef double min3(double a, double b, double c):
    cdef double m = a
    if b < m:
        m = b
    if c < m:
        m = c
    return m

cdef int imin3(int a, int b, int c):
    cdef int m = a
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

def edit_distance(np.ndarray[ITYPE_t,ndim=1] s, np.ndarray[ITYPE_t,ndim=1] t):
    cdef int n = s.shape[0]
    cdef int m = t.shape[0]
    cdef np.ndarray[ITYPE_t,ndim=2] ed = np.zeros((n+1,m+1), dtype=ITYPE)

    ed[:,0] = np.arange(n+1)
    ed[0,:] = np.arange(m+1)

    cdef unsigned int i,j

    for i in range(n):
        for j in range(m):
            if s[i] == t[j]:
                ed[i+1,j+1] = ed[i,j]
            else:
                ed[i+1,j+1] = imin3(ed[i,j+1] + 1, ed[i+1,j] + 1, ed[i,j] + 1)

    return ed[n,m]


def edit_distance_vc(np.ndarray[ITYPE_t,ndim=1] s, np.ndarray[ITYPE_t,ndim=1] t, int a, int b, int c):
    cdef int n = s.shape[0]
    cdef int m = t.shape[0]
    cdef np.ndarray[ITYPE_t,ndim=2] ed = np.zeros((n+1,m+1), dtype=ITYPE)

    ed[:,0] = np.arange(n+1)
    ed[0,:] = np.arange(m+1)

    cdef unsigned int i,j

    for i in range(n):
        for j in range(m):
            if s[i] == t[j]:
                ed[i+1,j+1] = ed[i,j]
            else:
                ed[i+1,j+1] = imin3(ed[i,j+1] + a, ed[i+1,j] + b, ed[i,j] + c)

    return ed[n,m]


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
