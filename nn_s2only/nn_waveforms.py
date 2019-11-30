
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

import argparse
import glob
import keras
import numpy as np
import os
import pickle
import psutil
import time

import keras
import utils_keras as kutils

from dataGenerator_waveforms import *

proc = psutil.Process(os.getpid())


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

class nn_waveforms():
    
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    def validate(self, mygen):
    
        print()
        
        ibatch = 0
        
        for x in mygen:
            
            x0 = x[0]
            x1 = x[1]
            
            # model prediction
            x2 = self.model.predict(x0)
            
            err = x1 - x2
            err_xmean = np.mean(err[:,0])
            err_ymean = np.mean(err[:,1])
                            
            print("mean x error: {0:.2f}".format(err_xmean))
            print("mean x error: {0:.2f}".format(err_ymean))
            
            ibatch += 1
            
            continue
       
    
        print("Batches: {0}".format(ibatch))
        
        #----------------------------------------------------------------------
        #----------------------------------------------------------------------
    
        return
    
    
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    def __init__(self):
        
        print(__name__ + "." + inspect.currentframe().f_code.co_name + "()")

        #----------------------------------------------------------------------
        #----------------------------------------------------------------------

        test_frac = 0.10
        
        self.events_per_file  = 1000
        self.args             = parse_arguments()
        self.events_per_batch = self.args.events_per_batch 
        self.downsample       = self.args.downsample
        self.downsample       = 100
        
        self.input_dim        = int(127000 / self.downsample)
        dir_data              = self.args.directory
        self.max_dirs         = self.args.max_dirs
        
        self.lst_dir_files    = glob.glob(dir_data + "/strArr*.npz")
        self.lst_dir_files.sort()
        self.lst_dir_files.sort(key=len)
        self.lst_dir_files    = self.lst_dir_files[:self.max_dirs]
        n_dir                 = len(self.lst_dir_files)
        n_test                = max(int(n_dir*test_frac), 1)
        n_train               = n_dir - n_test
        self.lst_files_train  = self.lst_dir_files[0:n_train]
        self.lst_files_test   = self.lst_dir_files[n_train:]
        self.n_events_train   = n_train*self.events_per_file
        self.n_epochs_train   = int( (self.events_per_file) / (self.events_per_batch) )*n_train
        self.arr2d_pred       = np.zeros(shape=(self.events_per_file*n_test, 6))

        self.model = None
        self.hist  = kutils.logHistory()
        self.t0    = time.time()

        
        #------------------------------------------------------------------------------
        #------------------------------------------------------------------------------
        
        print("Input Directory:  {0}".format(dir_data))
        print("Events:           {0}".format(self.n_events_train))
        print("Files Train:      {0}".format(n_train))
        print("Files Test:       {0}".format(n_test))
        print("Train files:")
        
        for x in (self.lst_files_train):
            print("   " + x)
        
        print("Test files:")
        
        for x in (self.lst_files_test):
            print("   " + x)
        
        assert(n_train > 0)
        assert(n_test > 0)
        assert(n_train + n_test == n_dir)
        assert(os.path.exists(dir_data))
        assert(1000 % self.downsample == 0)
        
        
        #----------------------------------------------------------------------
        #----------------------------------------------------------------------
        
        return

    
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    def run(self):
        
        self.train()
        #self.save()

        return
    
    
    #--------------------------------------------------------------------------
    # Virtual
    #--------------------------------------------------------------------------
    
    def train(self):
        return

    
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    def save(self):
    
        if (not self.history):
            print("No History")
            return
         
        #----------------------------------------------------------------------
        # Save.   To Do, Add: dropout rate, learning rate
        #----------------------------------------------------------------------
        
        acc         = int(100*np.round(self.history.history['acc'], 2))
        layers_desc = kutils.getModelDescription(self.model)
        f_model     = 'nn_modl_acc{0:.0f}_evts{1:05d}_{2}.h5'.format(acc, self.n_events_train, layers_desc)
        f_pred      = 'nn_pred_acc{0:.0f}_evts{1:05d}_{2}.npy'.format(acc, self.n_events_train, layers_desc)
        f_hist      = 'nn_hist_acc{0:.0f}_evts{1:05d}_{2}.npy'.format(acc, self.n_events_train, layers_desc)

        print("Saving '{0}'...".format(f_model))
        print("Saving '{0}'...".format(f_pred))
        
        self.model.save(f_model)                                  # Model
        np.save(f_pred.replace('.npy', ''), self.arr2d_pred)      # Predictions
        
        arrHist = self.hist.save(f_hist)
        
        # What about multiple Epochs?
        #print()
        #print("Batches:          {0}".format(self.hist.batches))
        #print("Events per Batch: {0}".format(self.events_per_batch))
        #print("Epochs:           {0}".format(self.n_epochs_train))
        #print("Events:           {0}".format(self.n_events_train))

        #assert(self.hist.batches == arrHist.shape[0])
        #assert(self.hist.batches*self.events_per_batch == self.n_events_train)
        
        self.t1 = time.time()
        dt = (self.t1 - self.t0)/60
    
        print("\nDone in {0:.1f} min".format(dt))
        
        
        
        #----------------------------------------------------------------------
        #----------------------------------------------------------------------
        
        return
    
    
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('-directory'       , required=True)
    parser.add_argument('-max_dirs'        , required=True, type=int)
    parser.add_argument('-events_per_batch', required=True, type=int)
    parser.add_argument('-samples'         , required=True, type=int)
    parser.add_argument('-downsample'      , required=True, type=int)
    
    arguments = parser.parse_args()
    
    return arguments



