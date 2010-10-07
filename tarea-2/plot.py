#!/usr/bin/env python

from __future__ import division
from numpy import *
import scipy.integrate
import pylab

def f(x):
    return 2 + 9 / (1 + 600 * (x - .43)**2)

def g(x):
    return exp(-x/4) * cos(2 * pi * f(x) * x)**2

x = linspace(0, 1, 2**20 + 1).astype(float32)

# compute integral using Simpson's rule
integral = scipy.integrate.simps(g(x), x)
print('{0:.5f}'.format(float(integral)))

# plot the function
pylab.plot(x, g(x))
pylab.show()

