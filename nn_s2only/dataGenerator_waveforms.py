
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

import inspect
import numpy as np
import keras
import os

from utils_python import *

import utils_keras as kutils
import utils_kernel as kernutils

proc = psutil.Process(os.getpid())


#******************************************************************************
#******************************************************************************

class DataGenerator_xy(keras.utils.Sequence):

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    def on_train_begin(self, logs=None):
        print("HERE")
        
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    def __init__(
        self,
        lst_files,
        events_per_file=1000,
        events_per_batch=100,
        n_inputs=127,
        n_outputs=2,
        downsample=1000,
        verbose=True
    ):
        
        #----------------------------------------------------------------------
        #----------------------------------------------------------------------

        self.lst_files        = lst_files
        self.n_files          = len(self.lst_files)
        self.events_per_file  = events_per_file
        self.events_per_batch = events_per_batch
        self.n_events         = int(self.n_files*self.events_per_file)
        self.n_batches        = int(self.n_events / self.events_per_batch)
        self.n_inputs         = n_inputs
        self.n_outputs        = n_outputs
        self.batch            = 0
        self.downsample       = downsample

        self.arr2d_waveforms  = np.zeros(shape=(self.events_per_batch, self.n_inputs))
        self.arr2d_xy         = np.zeros(shape=(self.events_per_batch, self.n_outputs))
        #self.n_epochs_train   = int( (self.events_per_file) / (self.events_per_batch) )*n_train

        assert(self.events_per_batch < self.events_per_file)
        
        
        #----------------------------------------------------------------------
        #----------------------------------------------------------------------
        
        if (verbose):
            #print("Input Directory:  {0}".format(dir_data))
            #print("Downsample:       {0}".format(self.downsample))
            #print("Input dimension:  {0}".format(self.input_dim))
            print("Batches:          {0}".format(self.n_batches))
            print("Events per batch: {0}".format(self.events_per_batch))
            print("Events:           {0}".format(self.n_events))
        
            print("\n{0} Input files:".format(len(self.lst_files)))
            for x in (self.lst_files):
                print("   " + x)
        
        
        #----------------------------------------------------------------------
        #----------------------------------------------------------------------
        
        return
    

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    def __getitem__(self, index):

        #------------------------------------------------------------------------------
        #------------------------------------------------------------------------------

        iFile = int(self.batch*self.events_per_batch / self.events_per_file)
        fpath = self.lst_files[iFile]
        sArr  = None
        i0    = int((self.batch*self.events_per_batch)%self.events_per_file)
        i1    = int(i0 + self.events_per_batch)
        j0    = i0 + iFile*self.events_per_file
        j1    = j0 + self.events_per_batch
        
        with np.load(fpath) as data:
            sArr = data['arr_0']
            
        arr3d             = sArr[:][:]['image']
        arr3d_ds          = kernutils.downsample_arr3d(arr3d, self.downsample)
        arr3d_batch       = arr3d_ds[i0:i1][:] 
        arr2d_batch       = arr3d_batch.reshape(arr3d_batch.shape[0], arr3d_batch.shape[1]*arr3d_batch.shape[2])
        arr2d_xy          = np.zeros(shape=(arr2d_batch.shape[0], 2))
        arr2d_xy[:, 0]    = sArr[i0:i1][:]['true_x']
        arr2d_xy[:, 1]    = sArr[i0:i1][:]['true_y']
        arr_s2areas_batch = sArr[i0:i1][:]['s2_areas']

        print("Batch: {0}/{1}, iFile {2}, i0={3:03d}, i1={4}, j0={5:03d}, j1={6}, Memory: {7} GB, File: {8}".format(
            self.batch,
            self.n_batches,
            iFile,
            i0,
            i1,
            j0,
            j1,
            getMemoryGB(proc),
            os.path.basename(fpath)
        ))

        
        #------------------------------------------------------------------------------
        #------------------------------------------------------------------------------

        self.batch += 1
        
        return arr2d_batch, arr2d_xy
        return self.arr2d_waveforms, self.arr2d_xy
        

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    def on_epoch_end(self):

        time.sleep(5) # Sometimes tensorflow print output lags
        
        print(__name__ + "." + inspect.currentframe().f_code.co_name + "()\n")
        print()
        
        print()
        
        return
    

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    def __len__(self):
        return 2
        #print(__name__ + "." + inspect.currentframe().f_code.co_name + "()")
        return self.n_batches # Batches per dataset/epoch
        
    
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------

#   arr2d_xy          = np.zeros(shape=(arr2d_batch.shape[0], 2))
#   arr2d_xy_s2       = np.zeros(shape=(arr2d_batch.shape[0], 2))
#   arr2d_xy[:, 0]    = sArr[i0:i1][:]['x_true']
#   arr2d_xy[:, 1]    = sArr[i0:i1][:]['y_true']
#   arr2d_xy_s2[:, 0] = sArr[i0:i1][:]['x_s2']
#   arr2d_xy_s2[:, 1] = sArr[i0:i1][:]['y_s2']
#   
#   print("\n   Evaluating Input Data: Predict batch {0}/{1}   (events: {2}-{3})".format(
#       ibatch+1,
#       self.n_batches,
#       i0,
#       i1
#   ))
#   
#   print("    -> Memory Usage: {0} GB".format(getMemoryGB(proc)))
#   
#   print("Predict...")
#   
#   arr2d_xy_pred = self.model.predict(arr2d_batch)
#  
#   self.arr2d_pred[i0:i1, 0] = arr2d_xy[:,0]
#   self.arr2d_pred[i0:i1, 1] = arr2d_xy[:,1]
#   self.arr2d_pred[i0:i1, 2] = arr2d_xy_s2[:,0]
#   self.arr2d_pred[i0:i1, 3] = arr2d_xy_s2[:,1]
#   self.arr2d_pred[i0:i1, 4] = arr2d_xy_pred[:,0]
#   self.arr2d_pred[i0:i1, 5] = arr2d_xy_pred[:,1]
#  
