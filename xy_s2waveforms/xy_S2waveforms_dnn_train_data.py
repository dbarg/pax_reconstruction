
####################################################################################################
####################################################################################################

import datetime
import math
import sys
import glob
import psutil
import os.path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd

from IPython.display import display

#sys.path.append(os.path.abspath("../../"))
#sys.path.append(os.path.abspath("../../pax_utils"))
#from s1s2_utils import *

from get_dnn_data import *


####################################################################################################
####################################################################################################

cols = [
    'event_number',
    'event_s2_count',
    'event_s2_length',
    'event_s2_left',
    'event_s2_right',
    'intr_count',
    'x',
    'y',
    'intr_x',
    'intr_y',
    'intr_z',
    'intr_drift_time',
    'intr_x_nn',
    'intr_y_nn'
]


####################################################################################################
####################################################################################################

#resample_factor = 2290
#resample_factor = 229
#resample_factor = 10
#resample_factor = 23
#resample_factor = 46
resample_factor = 230

iEventStart  = 0
nEventsTrain = 100 #25000
iEventEnd    = iEventStart + nEventsTrain

input_dir  = "../../pax_merge/merged/apr30/"
input_file = 'merged_all_200000.pkl'
input_path = input_dir + input_file
dir_in     = '../../pax_merge/merged/apr30/waveforms_s2waveforms_test_v2/new'


####################################################################################################
####################################################################################################


df_events     = pd.read_pickle(input_path)
df_events     = df_events[df_events['intr_count'] == 1].reset_index(drop=True)
s2_window_max = np.amax(df_events['event_s2_length'].as_matrix()  )
s2_window_max = int(math.ceil(s2_window_max / 100.0)) * 100

TotalEvents   = len(df_events.index)
df_events     = df_events[iEventStart:iEventEnd][:]
nTimesteps    = s2_window_max / resample_factor
df_events    = df_events[:][cols]


####################################################################################################
####################################################################################################

file_out_input = 'train/array_train_input_events%06d-%06d_timesteps%04d' % (iEventStart, iEventEnd - 1, nTimesteps)
file_out_truth = 'train/array_train_truth_events%06d-%06d_timesteps%04d' % (iEventStart, iEventEnd - 1, nTimesteps)

print()
print("Total Events:      " + str(TotalEvents))
print("Processing Events: " + str(nEventsTrain))
print("Start Event:       " + str(iEventStart))
print("End Event:         " + str(iEventEnd))
print("Shape:             " + str(df_events.shape))
print("Max S2 Window:     " + str(s2_window_max))
print("file_out_input:    " + file_out_input)
print("file_out_truth:    " + file_out_truth)
print()

assert(s2_window_max % resample_factor == 0)

#display(df_events[0:5][:])
#exit(1)

    
####################################################################################################
# Training Data
####################################################################################################
 
train_data, train_truth  = get_data(dir_in, df_events, s2_window_max, resample_factor)

if (resample_factor > 1):
    
    s2_window_max = int(s2_window_max / resample_factor)
    
print()
print("Input data shape:       " + str(train_data.shape ))
print("Truth data shape:       " + str(train_truth.shape))
print()


####################################################################################################
####################################################################################################

np.save(file_out_input, train_data)
np.save(file_out_truth, train_truth)

test_train_data  = np.load(file_out_input + '.npy')
test_train_truth = np.load(file_out_truth + '.npy')

print(test_train_data.shape)
print(test_train_truth.shape)
print()
