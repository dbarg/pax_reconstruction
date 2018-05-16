
####################################################################################################
####################################################################################################

import argparse
import datetime
import glob
import json
import os
import pprint
import sys
import time

import numpy as np
import pandas as pd
import scipy as sp

import keras
from keras import layers
from keras.callbacks import CSVLogger
from keras.models import load_model
from keras.utils import plot_model

sys.path.append("/home/dbarge")
from keras_utils.config_utils import *
from keras_utils.dnn_utils import *

pp = pprint.PrettyPrinter(depth=4)


####################################################################################################
####################################################################################################

def main(
    file_input,
    file_truth,
    n_timesteps,
    n_outputs,
    layers_hidden,
    n_events_train,
    n_epochs,
    loss,
    optimizer):

    printVersions()

    
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
    
    model = dnnModel(
        n_inputs,
        n_outputs,
        layers_hidden,
        'elu',
        loss,
        optimizer,
        keep_rate=0.00005)
    
    print()
    print("Model Summary:")
    model.summary()
    print()
    
    #print(model.get_layer('input'))
    #print(model.get_layer('encoded'))
    
    #return

    ######################################################################################
    # Fit Model
    #
    #  to do: reset model 
    ######################################################################################
    
    history = model.fit(
        train_data,
        train_truth,
        batch_size=64,
        epochs=n_epochs,
        verbose=2
    )
    
    
    ######################################################################################
    ######################################################################################
    
    dct_history = history.history
    
    last_loss = dct_history['loss'][n_epochs-1]
    last_loss = int(round(last_loss*100, 0))
    last_loss = 'loss' + str(last_loss)
    
    acc  = dct_history['acc'][n_epochs-1]
    acc  = int(round(acc*1e4, 0))
    acc  = 'ac%04d' % acc
    
    
    ######################################################################################
    ######################################################################################
    
    config     = model.get_config()
    #layer_desc = getModelDescription(config)   
    
    #pp.pprint(config)
    
    
    
    ######################################################################################
    # Save Model
    ######################################################################################
    
    loss_desc = loss
    
    if ('_' in loss_desc):
        
        loss_desc = ''.join( [x[:1] for x in loss.split('_') ] )
    

    folder   = "models/"    
    desc     = 'dnn_' + model_name + ("_ts%04d" % n_timesteps) + '_' + ('e%02d' % n_epochs) + '_' + loss_desc + '_' + optimizer + '_' + acc + '_' + layers_desc
    
    name_h5  = folder + desc + '.h5'
    name_cfg = folder + desc + '.json'
    name_png = folder + desc + '.png'
    
    
    ######################################################################################
    # Save
    ######################################################################################
    
    model.save(name_h5, overwrite=True)
    with open(name_cfg, 'w') as fp: json.dump(config, fp)
    plot_model(model, to_file=name_png, show_layer_names=True, show_shapes=True)
        
    print("\nSaved model: '" + name_h5 + "'\n")
    
    
    ######################################################################################
    ######################################################################################

    return



####################################################################################################
####################################################################################################

if __name__ == "__main__":

    t0 = time.time()
    
    ################################################################################################
    ################################################################################################
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-file_input'    , required=True)
    parser.add_argument('-file_truth'    , required=True)
    parser.add_argument('-n_timesteps'   , required=True, type=int)
    parser.add_argument('-n_outputs'     , required=True, type=int)
    parser.add_argument('-n_events_train', required=True, type=int)
    parser.add_argument('-n_epochs'      , required=True, type=int)
    parser.add_argument('-layers_hidden' , required=True, type=int, nargs="+")

    parser.add_argument('-loss'          , required=True)
    parser.add_argument('-optimizer'     , required=True)

        
    args = parser.parse_args()

    file_input     = args.file_input
    file_truth     = args.file_truth
    n_timesteps    = args.n_timesteps
    n_outputs      = args.n_outputs
    n_events_train = args.n_events_train
    n_epochs       = args.n_epochs
    
    layers_hidden  = args.layers_hidden

    loss           = args.loss
    optimizer      = args.optimizer

    print()
    print("file_input:    " + str(file_input) )
    print("file_truth:    " + str(file_truth) )
    print()

    assert(os.path.exists(file_input))
    assert(os.path.exists(file_truth))

    
    ################################################################################################
    ################################################################################################
    
    main(file_input,
         file_truth,
         n_timesteps,
         n_outputs,
         layers_hidden,
         n_events_train,
         n_epochs,
         loss,
         optimizer)

    
    ################################################################################################
    ################################################################################################
    
    t1 = time.time()
    dt = round(t1 - t0, 0)
    dt = datetime.timedelta(seconds=dt)
    print("Total Time: " + str(dt))


