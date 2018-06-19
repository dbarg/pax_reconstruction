
####################################################################################################
####################################################################################################

import argparse
import datetime
import glob
import json
import os
import pprint
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import scipy as sp

import keras
from keras import layers
from keras import models

from keras.callbacks import CSVLogger
from keras.models import load_model
from keras.utils import plot_model

sys.path.append("/home/dbarge")
from keras_utils.config_utils import *
from keras_utils.dnn_utils import *

pp = pprint.PrettyPrinter(depth=4)



    
####################################################################################################
####################################################################################################

def main():
    
    file_input = '/scratch/midway2/dbarge/train_pax65/array_train_input_events000000-199999_timesteps0010.npy'
    file_truth = '/scratch/midway2/dbarge/train_pax65/array_train_truth_events000000-199999_timesteps0010.npy'
    
    name_h5    = 'sbatch/models/cpu/dnn_s2waveforms-xy_ts0010_e50_mse_adam_ac9947_layers1270-1270-127-2.h5'
    
    
    ######################################################################################
    # Predict
    ######################################################################################
    
    n_events_train = 100000
    
    train_data  = np.load(file_input)
    train_truth = np.load(file_truth)
    
    test_data  = train_data [n_events_train:, :]
    test_truth = train_truth[n_events_train:, :]

    train_data  = train_data [0:n_events_train, :]
    train_truth = train_truth[0:n_events_train, :]
    
    print("Train Input shape: " + str(train_data.shape ))
    print("Train Truth shape: " + str(train_truth.shape))
    print("Test Input shape:  " + str(test_data.shape ))
    print("Test Truth shape:  " + str(test_truth.shape))

    model    = load_model(name_h5)
        
    arr_pred = model.predict(test_data)

    print(arr_pred.shape)
    
    
    ####################################################################################################
    ####################################################################################################
    
    df_out           = pd.DataFrame()
    
    df_out['x_pred'] = arr_pred[:, 0]
    df_out['y_pred'] = arr_pred[:, 1]
    
    df_out['x_true'] = test_truth[:, 0]
    df_out['y_true'] = test_truth[:, 1]

    dir_pred = "sbatch/predictions/cpu/"
    
    file_hdf = dir_pred + os.path.basename(name_h5).replace('.h5', '.hdf5')
    #file_hdf = 'test.hdf5'

    df_out.to_hdf(file_hdf, 'df')

    print("\nSaved predictions: '" + file_hdf + "'\n")


    
    ######################################################################################
    ######################################################################################

    return



####################################################################################################
####################################################################################################

if __name__ == "__main__":

    main()
