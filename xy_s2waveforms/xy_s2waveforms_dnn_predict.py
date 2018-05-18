
####################################################################################################
####################################################################################################

import sys
import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

import keras
import keras.utils.vis_utils
from keras import backend as K
from keras import layers
from keras.layers import Dense
from keras.models import load_model
from keras.models import Sequential
from keras.utils import plot_model

sys.path.append(os.path.abspath("../"))
from helpers import *

#sys.path.append(os.path.abspath("../../"))
#from pax_utils import s1s2_utils

from get_dnn_data import *

print(pd.__version__)


####################################################################################################
####################################################################################################
    
s2_window_max   = 2300
resample_factor = 46
resample_factor = 230

n_timesteps     = int(s2_window_max / resample_factor)

assert(s2_window_max % resample_factor == 0)

file_out_input = '/scratch/midway2/dbarge/train_pax65/'

#file_out_input += 'array_train_input_events000000-199999_timesteps0010.npy'
file_out_input += 'array_train_input_events000000-199999_timesteps0020.npy'
#file_out_input += 'array_train_input_events000000-199999_timesteps0025.npy'
#file_out_input += 'array_train_input_events000000-199999_timesteps0050.npy'
#file_out_input += 'array_train_input_events000000-199999_timesteps0100.npy'

file_out_truth = file_out_input.replace('input', 'truth')

nEventsTrain = 100000


####################################################################################################
####################################################################################################

model_name_h5 = './sbatch/models/'

model_name_h5 += 'dnn_s2waveforms-xy_ts0020_e20_mse_adam_ac9902_layers2540-2540-1270-127-2.h5'
model_name_h5 += 'dnn_s2waveforms-xy_ts0010_e10_mse_adam_ac9912_layers1270-1270-127-2.h5'

file_pkl = './predictions/' + os.path.basename(model_name_h5).replace('.h5', '.pkl')
file_hdf = './predictions/' + os.path.basename(model_name_h5).replace('.h5', '.hdf5')

print(file_hdf)


####################################################################################################
# Load Training Data
####################################################################################################

train_data  = np.load(file_out_input)
train_truth = np.load(file_out_truth)

train_data  = train_data [nEventsTrain:, :]
train_truth = train_truth[nEventsTrain:, :]

print()
print(train_data.shape)
print(train_truth.shape)
print()




####################################################################################################
# Predict
####################################################################################################

model       = load_model(model_name_h5)
arr_xy_pred = model.predict(train_data)

print("\nLoaded Model: " + model_name_h5)
print("Predicted\n")



####################################################################################################
####################################################################################################

arr_x_true = train_truth[:, 0]
arr_y_true = train_truth[:, 1]

arr_x_pred = arr_xy_pred[:, 0]
arr_y_pred = arr_xy_pred[:, 1]

arr_dx     = arr_x_true - arr_x_pred
arr_dy     = arr_y_true - arr_y_pred

print(arr_x_pred.shape)
print(arr_y_pred.shape)
print(arr_x_true.shape)
print(arr_y_true.shape)


####################################################################################################
####################################################################################################

df_out = pd.DataFrame()

df_out['x_pred'] = arr_x_pred
df_out['y_pred'] = arr_y_pred

df_out['x_true'] = arr_x_true
df_out['y_true'] = arr_y_true


df_out.to_pickle(file_pkl, protocol=3)
df_out.to_hdf(file_hdf, 'df')


####################################################################################################
####################################################################################################

df_hdf = pd.read_hdf(file_hdf)

display(df_hdf[0:5][:])
print()
