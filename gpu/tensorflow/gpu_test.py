####################################################################################################
####################################################################################################

import os
import sys
import time

import numpy

import tensorflow as tf
#import keras

print()
print("Python Version:     " + sys.version)
print("Tensorflow Version: " + tf.__version__)
#print("Keras Version:      " + keras.__version__)
print()



####################################################################################################
####################################################################################################

t0 = time.time()

# Creates a graph.
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)

# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# Runs the op.
print(sess.run(c))

t1 = time.time()
dt = round(t1 - t0, 0)
print("\ndt: " + str(dt) + " s\n")


####################################################################################################
####################################################################################################


