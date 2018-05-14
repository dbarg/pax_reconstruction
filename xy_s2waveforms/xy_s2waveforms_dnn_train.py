
####################################################################################################
####################################################################################################

import argparse
import datetime
import glob
import json
import os
import pprint
import sys

import numpy as np
import pandas as pd
import scipy as sp

#from PIL import Image

#import theano
#import pygpu

#import tensorflow 

import keras
#from keras import backend as K
from keras import layers
from keras.models import load_model
#from keras.utils import plot_model

#sys.path.append(os.path.abspath("../../"))
sys.path.append("/home/dbarge")
#sys.path.append(os.path.abspath("~/keras_utils"))

print("\n" + str(sys.path) + "\n")

from keras_utils.config_utils import *
from keras_utils.dnn import *

pp = pprint.PrettyPrinter(depth=4)

ver       = keras.__version__.split('.')
ver_major = int(ver[0])


####################################################################################################
####################################################################################################

def main(
    file_input,
    file_truth,
    n_timesteps,
    n_outputs,
    layers_hidden,
    n_events_train):


    ################################################################################################
    ################################################################################################

    model_name  = 's2waveforms-xy'

    n_channels  = 127
    n_inputs    = n_timesteps * n_channels
    layers_desc = 'layers' + str(n_inputs) + '-' + str('-').join(str(x) for x in layers_hidden) + '-' + str(n_outputs)


    ####################################################################################################
    ####################################################################################################
    
    train_data  = np.load(file_input)
    train_truth = np.load(file_truth)
    
    train_data  = train_data [0:n_events_train, :]
    train_truth = train_truth[0:n_events_train, :]
    
    
    ####################################################################################################
    # Training Data
    ####################################################################################################
    
    print()
    print("Channels:         " + str(n_channels))
    print("Timesteps:        " + str(n_timesteps))
    print("Outputs:          " + str(n_outputs) )
    print("Input data shape: " + str(train_data.shape ))
    print("Truth data shape: " + str(train_truth.shape))
    print()
    
    
    ####################################################################################################
    # Initialize Model
    ####################################################################################################
    
    model = dnnModel(n_inputs, n_outputs, layers_hidden, 'elu', 0.00005)
    
    print()
    print("Model Summary:")
    model.summary()
    print()
    
    #print(model.get_layer('input'))
    #print(model.get_layer('encoded'))
    
    
    ######################################################################################
    # Fit Model
    ######################################################################################
    
    # to do: reset model 
    
    epochs = 1
    
    if (ver_major >= 2):
        
        history = model.fit(
            train_data,
            train_truth,
            batch_size=64,
            epochs=epochs,
            verbose=True
        )
    
    else:
    
        history = model.fit(
            train_data,
            train_truth,
            batch_size=64,
            #epochs=epochs,
            nb_epoch=epochs,
            verbose=True
        )
    
    
    ######################################################################################
    ######################################################################################
    
    dct_history = history.history
    
    loss = dct_history['loss'][epochs-1]
    loss = int(round(loss*100, 0))
    loss = 'loss' + str(loss)
    
    acc  = dct_history['acc'][epochs-1]
    acc  = int(round(acc*1e4, 0))
    acc  = 'acc%04d' % acc
    
    
    ######################################################################################
    ######################################################################################
    
    config     = model.get_config()
    #layer_desc = getModelDescription(config)   
    
    #pp.pprint(config)
    
    
    ######################################################################################
    # Save Model
    ######################################################################################
    
    folder   = "models/"    
    desc     = 'dnn_' + model_name + '_' + acc + '_epochs' + str(epochs) + '_' + layers_desc 
    name_h5  = folder + desc + '.h5'
    name_png = name_h5.replace('.h5', '.png')
    name_cfg = name_h5.replace('.h5', '.json')
    
    
    ######################################################################################
    # Save
    ######################################################################################
    
    print("\nSaving model: '" + name_h5 + "'\n")
    
    model.save(name_h5, overwrite=True)
    with open(name_cfg, 'w') as fp: json.dump(config, fp)
    #plot_model(model, to_file=name_png, show_layer_names=True, show_shapes=True)


####################################################################################################
####################################################################################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-file_input'    , required=True)
    parser.add_argument('-file_truth'    , required=True)
    parser.add_argument('-n_timesteps'   , required=True, type=int)
    parser.add_argument('-n_outputs'     , required=True, type=int)
    parser.add_argument('-n_events_train', required=True, type=int)
    parser.add_argument('-layers_hidden' , required=True, type=int, nargs="+")
    
    args = parser.parse_args()

    file_input     = args.file_input
    file_truth     = args.file_truth
    n_timesteps    = args.n_timesteps
    n_outputs      = args.n_outputs
    n_events_train = args.n_events_train
    layers_hidden  = args.layers_hidden

    print()
    print("file_input:    " + str(file_input) )
    print("file_truth:    " + str(file_truth) )
    print()

    assert(os.path.exists(file_input))
    assert(os.path.exists(file_truth))

    main(file_input, file_truth, n_timesteps, n_outputs, layers_hidden, n_events_train)

