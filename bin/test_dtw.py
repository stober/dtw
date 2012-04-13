#! /usr/bin/env python
"""
Author: Jeremy M. Stober
Program: TEST_DTW.PY
Date: Wednesday, March  7 2012
Description: Test DTW algorithms.
"""

import numpy as np
from dtw import *
import dtw.fast
import numpy.random as npr
import pylab

if __name__ == '__main__':

    import itertools

    # create sequences of related sequences
    x = npr.normal(0,15,(10,2))
    e = npr.normal(0,1,(10,2))
    y = x + e

    xa = []
    t = np.array([0.0,0.0])
    for i in x:
        t += i
        xa.append(t.copy())

    ya = []
    t = np.array([0.0,0.0])
    for i in x + e:
        t += i
        ya.append(t.copy())


    xa = np.array(xa)
    ya = np.array(ya)

    pylab.plot(xa[:,0],xa[:,1])
    pylab.plot(ya[:,0],ya[:,1])
    print "Slow Version"
    print dtw_distance(xa,ya) #,[1.0,1.0,0.0])
    print "Fast Version"
    print dtw.fast.dtw_fast(xa,ya)
    pylab.show()



