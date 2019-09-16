
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

import keras
import numpy
import os
import pygpu
import sys
import theano
import time

from theano import config
from theano import function
from theano import shared
from theano import tensor


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


print("Python Version: {0}".format(sys.version))
print("Theano Version: {0}".format(theano.version))
print("Keras Version:  {0}".format(keras.version))

pygpu.test()
theano.test()


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
iters = 1000

rng = numpy.random.RandomState(22)
x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
f = function([], tensor.exp(x))
t0 = time.time()

for i in range(iters):
    r = f()
    continue
    
t1 = time.time()


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

print(f.maker.fgraph.toposort())
print("Looping %d times took %f seconds" % (iters, t1 - t0))
print("Result is %s" % (r,))

if numpy.any([isinstance(x.op, tensor.Elemwise) and ('Gpu' not in type(x.op).__name__) for x in f.maker.fgraph.toposort()]):
    print('Used the cpu')
    
else:
    print('Used the gpu')
