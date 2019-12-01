
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

from generator_waveforms import *

proc = psutil.Process(os.getpid())


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

class nn_waveforms():
    
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    def reco(self, mygen):
        
        return
    
    
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    def validate(self, mygen):

        print("\n\n----- Validate -----\n\n")
    
        #----------------------------------------------------------------------
        #----------------------------------------------------------------------
        
        self.strArrPred = np.zeros(
            self.n_events_test,
            dtype=[
                ('x_true', np.float32),
                ('y_true', np.float32),
                ('x_pred', np.float32),
                ('y_pred', np.float32),
                ('x_reco', np.float32),
                ('y_reco', np.float32)
            ]
        )
            
        
        #----------------------------------------------------------------------
        #----------------------------------------------------------------------

        ibatch = 0
        
        for x in mygen:
            
            i0 = ibatch*self.events_per_batch
            i1 = i0 + self.events_per_batch
            
            x_in    = x[0]
            xy      = x[1]
            xy_pred = self.model.predict(x_in)
            
            print(i0)
            
            self.strArrPred[i0:i1]['x_true'] = xy[:, 0]
            self.strArrPred[i0:i1]['y_true'] = xy[:, 1]
            self.strArrPred[i0:i1]['x_pred'] = xy_pred[:, 0]
            self.strArrPred[i0:i1]['y_pred'] = xy_pred[:, 1]
            self.strArrPred[i0:i1]['x_reco'] = xy[:, 2]
            self.strArrPred[i0:i1]['y_reco'] = xy[:, 3]
            
            ibatch += 1
            
            continue
        
        #----------------------------------------------------------------------
        #----------------------------------------------------------------------

        
        print("Batches: {0}".format(ibatch))
    
        return
    
    
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    def __init__(self):
        
        print(__name__ + "." + inspect.currentframe().f_code.co_name + "()")

        #----------------------------------------------------------------------
        #----------------------------------------------------------------------
        
        self.events_per_file  = 1000
        self.args             = parse_arguments()
        dir_data              = self.args.directory
        self.max_dirs         = self.args.max_dirs
        
        self.events_per_batch = self.args.events_per_batch 
        self.downsample       = self.args.downsample
        self.input_dim        = int(127000 / self.downsample)
        
        self.lst_dir_files    = glob.glob(dir_data + "/strArr*.npz")
        self.lst_dir_files.sort()
        self.lst_dir_files.sort(key=len)
        self.lst_dir_files    = self.lst_dir_files[:self.max_dirs]
        n_dir                 = len(self.lst_dir_files)
        test_frac             = 0.10
        n_test                = max(int(n_dir*test_frac), 1)
        n_train               = n_dir - n_test
        self.lst_files_train  = self.lst_dir_files[0:n_train]
        self.lst_files_test   = self.lst_dir_files[n_train:]
        self.n_events_train   = n_train*self.events_per_file
        self.n_events_test    = n_test*self.events_per_file
        self.dir_out          = './models/'

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
        
        print(n_train)
        assert(n_train > 0)
        assert(n_test > 0)
        assert(n_train + n_test == n_dir)
        assert(os.path.exists(dir_data))
        assert(1000 % self.downsample == 0)
        assert(self.input_dim % 127 == 0)
        
        #----------------------------------------------------------------------
        #----------------------------------------------------------------------
        
        return

    
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    def run(self):
        
        self.train()
        self.save()

        return
    
    
    #--------------------------------------------------------------------------
    # Virtual
    #--------------------------------------------------------------------------
    
    def train(self):
        print("Virtual")
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
        f_model     = self.dir_out + 'nn_modl_acc{0:.0f}_evts{1:05d}_{2}.h5'.format(acc, self.n_events_train, layers_desc)
        f_pred      = self.dir_out + 'nn_pred_acc{0:.0f}_evts{1:05d}_{2}.npy'.format(acc, self.n_events_train, layers_desc)
        f_hist      = self.dir_out + 'nn_hist_acc{0:.0f}_evts{1:05d}_{2}.npy'.format(acc, self.n_events_train, layers_desc)

        print()
        print("Saving '{0}'...".format(f_model))
        print("Saving '{0}'...".format(f_pred))
        print("Saving '{0}'...".format(f_hist))
        print()
        
        self.model.save(f_model)
        np.save(f_pred.replace('.npy', ''), self.strArrPred)
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
    parser.add_argument('-downsample'      , required=True, type=int)
    
    arguments = parser.parse_args()
    
    return arguments



