
####################################################################################################
####################################################################################################

import datetime
import sys
import glob
import psutil
import os.path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd

from IPython.display import display

sys.path.append(os.path.abspath("../../"))
sys.path.append(os.path.abspath("../../pax_utils"))
from s1s2_utils import *

#from model_xy_s2waveforms_dnn import *
from get_dnn_data import *


####################################################################################################
####################################################################################################

#nEventsTrain = 200000
#nEventsTrain = 100000
#nEventsTrain = 10000
nEventsTrain = 50000
#nEventsTrain = 100

input_dir  = "../../pax_merge/merged/apr30/"
input_file = 'merged_all_200000.pkl'
input_path = input_dir + input_file


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
    'intr_y']

df_events, s2_window_max = getEventsDataFrame(input_path)
df_events                = df_events[df_events['intr_count'] == 1].reset_index(drop=True)
df_events                = df_events[0:nEventsTrain][:]
#df_events                = df_events[:][cols]


####################################################################################################
####################################################################################################

print(df_events.shape)
display(df_events[0:5][:])

print()
print("Event Max S2 Window size: " + str(s2_window_max))
print()
        

    
####################################################################################################
# Training Data
####################################################################################################

dir_in = '../../pax_merge/merged/apr30/waveforms_s2waveforms_test_v2/new'

#resample_factor = 2290
#resample_factor = 229
resample_factor = 10

assert(s2_window_max % resample_factor == 0)



####################################################################################################
####################################################################################################
 
train_data, train_truth  = get_data(dir_in, df_events, s2_window_max, resample_factor)

if (resample_factor > 1):
    
    s2_window_max = int(s2_window_max / resample_factor)
    
    
print()
print("Input data shape:       " + str(train_data.shape ))
print("Truth data shape:       " + str(train_truth.shape))
print()

    