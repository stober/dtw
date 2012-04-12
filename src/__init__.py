#! /usr/bin/env python
"""
Author: Jeremy M. Stober
Program: DTW.PY
Date: Tuesday, February 15 2011
Description: Dynamic Time Warping
"""

import numpy as np
import numpy.linalg as la

def equalize(s,t,default=None):
    # equalize the length of two strings by appending a default value

    if len(s) < len(t):
        s.extend([default] * (len(t) - len(s)))
    elif len(t) < len(s):
        t.extend([default] * (len(s) - len(t)))
    else:
        pass # same length

    return s,t

def non_dtw_distance(s,t,default=None,costf=None):
    # Don't run dynamic time warping, instead just compare actions
    # using the provided cost function and post-pend to make the
    # sequences the same length.

    s,t = equalize(s,t,default)

    return sum([costf(a,b) for a,b in zip(s,t)])

def initialize_dmatrix(rows,cols):
    d = np.zeros((rows,cols),dtype='float')

    for i in range(rows):
        d[i,0] = 1e6
    for j in range(cols):
        d[0,j] = 1e6

    d[0,0] = 0

    return d



# Note that wikipedia orginally recommended initializing using Inf
# values. Ex:

#d[:,0] = 1e6
#d[0,:] = 1e6
#d[0,0] = 0

#This means that comparing to any empty string would result
# in an Inf value. Need to reason about boundry conditions.



def edit_distance(s,t):
    n = len(s)
    m = len(t)
    d = initialize_dmatrix(n+1,m+1)

    for i in range(1,n+1):
        for j in range(1,m+1):
            if s[i-1] == t[j-1]:
                d[i,j] = d[i-1,j-1]
            else:
                d[i,j] = min(d[i-1,j] + 1, d[i,j-1] + 1, d[i-1,j-1] + 1)

    return d[n,m]

def etw_distance(list1, list2, params, costf=lambda x,y: la.norm(x - y)):
    """
    etw_distance : extended time warping
    Use dynamic time warping but apply a cost to (insertion, deletion, match)
    """

    n = len(list1)
    m = len(list2)
    dtw = initialize_dmatrix(n+1,m+1)

    icost = params[0]
    dcost = params[1]
    mcost = params[2]

    for (i,x) in enumerate(list1):
        i += 1
        for (j,y) in enumerate(list2):
            j += 1

            cost = costf(x,y)
            dtw[i,j] = cost + min(dtw[i-1,j] + icost, dtw[i,j-1] + dcost, dtw[i-1][j-1] + mcost)

    return dtw[n,m]

def dtw_distance(list1, list2, costf=lambda x,y: la.norm(x - y) ):

    n = len(list1)
    m = len(list2)
    dtw = initialize_dmatrix(n+1,m+1)

    for (i,x) in enumerate(list1):
        i += 1
        for (j,y) in enumerate(list2):
            j += 1

            cost = costf(x,y)
            dtw[i,j] = cost + min(dtw[i-1,j],dtw[i,j-1],dtw[i-1][j-1])

    return dtw[n,m]

def dtw_wdistance(list1, list2, w, costf=lambda x,y: la.norm(x - y)):

    n = len(list1)
    m = len(list2)
    dtw = initialize_dmatrix(n+1,m+1)

    for (i,x) in enumerate(list1):
        for j in range(max(0,i-w),min(m,i+w)):
            y = list2[j]
            cost = costf(x,y)
            dtw[i,j] = cost + min(dtw[i-1,j],dtw[i,j-1],dtw[i-1][j-1])

    return dtw[n-1,m-1]




