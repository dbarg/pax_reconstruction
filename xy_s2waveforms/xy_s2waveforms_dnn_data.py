
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

import datetime
import math
import numpy as np
import pandas as pd
import sys
import glob
import os.path

from get_dnn_data import *


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#
#cols = [
#    'event_number',
#    'event_s2_count',
#    'event_s2_length',
#    'event_s2_left',
#    'event_s2_right',
#    'intr_count',
#    'x',
#    'y',
#    'intr_x',
#    'intr_y',
#    'intr_z',
#    'intr_drift_time',
#    'intr_x_nn',
#    'intr_y_nn'
#]


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#iEventStart  = 0
#nEventsTrain = 200000
#iEventEnd    = iEventStart + nEventsTrain
##input_dir  = "/home/dbarge/scratch/simulations/wimp/merged/may07/"
##input_file = 'merged_all_200000.pkl'
##input_path = input_dir + input_file
##dir_in     = '/home/dbarge/scratch/simulations/wimp/merged/may07/waveforms_s2waveforms_v2/s2/'
#input_dir  = "/project/lgrandi/dbarge/simulation/wimp/pax_v6.8.3/er/merged/"
#input_file = 'merged_all_200000.pkl'
#input_path = input_dir + input_file
#dir_in     = '/project/lgrandi/dbarge/simulation/wimp/pax_v6.8.3/er/merged/waveforms_s2waveforms_v2/s2/'


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#df_events     = pd.read_pickle(input_path)
#df_events     = df_events[df_events['intr_count'] == 1].reset_index(drop=True)
##s2_window_max = np.amax(df_events['event_s2_length'].as_matrix()  )
#s2_window_max = np.amax(df_events['event_s2_length'].values )
#s2_window_max = int(math.ceil(s2_window_max / 100.0)) * 100
#TotalEvents   = len(df_events.index)
#df_events     = df_events[iEventStart:iEventEnd][:]


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#assert (s2_window_max < 2000)
#s2_window_max = 2000
#
#resample_factor = 200 # 10
#
##resample_factor = 1900 # 1
##resample_factor = 190 # 10
##resample_factor = 100  # 19
##resample_factor = 76  # 25
##resample_factor = 50  # 38
#
#nTimesteps    = s2_window_max / resample_factor
#df_events    = df_events[:][cols]
#    
#print("S2 Window Max: " + str(s2_window_max))
#print("Timesteps:     " + str(nTimesteps))
#print(TotalEvents)



#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#file_out_input = 'train_69/array_train_input_events%06d-%06d_timesteps%04d' % (iEventStart, iEventEnd - 1, nTimesteps)
#file_out_truth = 'train_69/array_train_truth_events%06d-%06d_timesteps%04d' % (iEventStart, iEventEnd - 1, nTimesteps)

dir_in         = '/project/lgrandi/dbarge/simulation/wimp/pax_v6.8.3/er/merged/waveforms_s2waveforms_v2/s2/'
#file_out_input = 'train_683/array_train_input_events%06d-%06d_timesteps%04d' % (iEventStart, iEventEnd - 1, nTimesteps)
#file_out_truth = 'train_683/array_train_truth_events%06d-%06d_timesteps%04d' % (iEventStart, iEventEnd - 1, nTimesteps)


#print()
#print("Total Events:      " + str(TotalEvents))
#print("Processing Events: " + str(nEventsTrain))
#print("Start Event:       " + str(iEventStart))
#print("End Event:         " + str(iEventEnd))
#print("Shape:             " + str(df_events.shape))
#print("Max S2 Window:     " + str(s2_window_max))
#print("file_out_input:    " + file_out_input)
#print("file_out_truth:    " + file_out_truth)
#print()
#
#assert(s2_window_max % resample_factor == 0)

#display(df_events[0:5][:])
#exit(1)

    
#------------------------------------------------------------------------------
# Training Data
#------------------------------------------------------------------------------
 
resample_factor = 200

train_data, train_truth  = get_data(dir_in, df_events, resample_factor)

if (resample_factor > 1):
    
    s2_window_max = int(s2_window_max / resample_factor)
    
print()
print("Input data shape:       " + str(train_data.shape ))
print("Truth data shape:       " + str(train_truth.shape))
print()

#exit(1)


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#np.save(file_out_input, train_data)
#np.save(file_out_truth, train_truth)

#test_train_data  = np.load(file_out_input + '.npy')
#test_train_truth = np.load(file_out_truth + '.npy')

#print(test_train_data.shape)
#print(test_train_truth.shape)
#print()
