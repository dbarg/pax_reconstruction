
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

class DataGenerator(keras.utils.Sequence):
        
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    def __init__(
        self,
        lst_files,
        events_per_file=1000,
        events_per_batch=100,
        n_inputs=127,
        n_outputs=2,
        downsample=10,
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

        assert(self.events_per_batch < self.events_per_file)
        
        
        #----------------------------------------------------------------------
        #----------------------------------------------------------------------
        
        if (verbose):
            #print("Input Directory:  {0}".format(dir_data))
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

        self.batch = index
        
        #------------------------------------------------------------------------------
        #------------------------------------------------------------------------------

        sArr  = None
        iFile = int(self.batch*self.events_per_batch / self.events_per_file)
        fpath = self.lst_files[iFile]
        i0    = int((self.batch*self.events_per_batch)%self.events_per_file)
        i1    = int(i0 + self.events_per_batch)
        j0    = i0 + iFile*self.events_per_file
        j1    = j0 + self.events_per_batch
        
        with np.load(fpath) as data:
            sArr = data['arr_0']
            sArr = sArr[i0:i1][:][:]
            
        arr3d         = sArr[:][:]['image']
        arr3d_ds      = kernutils.downsample_arr3d(arr3d, self.downsample)
        arr3d_batch   = arr3d_ds[:][:] 
        arr2d_batch   = arr3d_batch.reshape(arr3d_batch.shape[0], arr3d_batch.shape[1]*arr3d_batch.shape[2])
        
        #arr2d_xy      = np.zeros(shape=(arr2d_batch.shape[0], 2))
        #arr2d_xy[:,0] = sArr[:]['true_x']
        #arr2d_xy[:,1] = sArr[:]['true_y']

        #print("Index: {0}".format(index))
        print("-> Data Generator Batch: {0}/{1}, iFile {2}, i0={3:03d}, i1={4}, j0={5:03d}, j1={6}, Memory: {7} GB, File: {8}".format(
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

        self.batch += 1
        
        #------------------------------------------------------------------------------
        #------------------------------------------------------------------------------
        
        return arr2d_batch, sArr
        #return arr2d_batch, arr2d_xy
        

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    def on_epoch_end(self):

        #time.sleep(5) # Sometimes tensorflow print output lags
        print(__name__ + "." + inspect.currentframe().f_code.co_name + "()\n")
        print()
        
        return
    

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    def __len__(self):
        #return 2
        #print(__name__ + "." + inspect.currentframe().f_code.co_name + "()")
        return self.n_batches # Batches per dataset/epoch

    
#******************************************************************************
#******************************************************************************

class DataGenerator_xy(DataGenerator):
    
    def __getitem__(self, index):
    
        arr2d_batch, sArr =  super(DataGenerator_xy, self).__getitem__(index)

        arr2d_xy      = np.zeros(shape=(arr2d_batch.shape[0], 2))
        arr2d_xy[:,0] = sArr[:]['true_x']
        arr2d_xy[:,1] = sArr[:]['true_y']
        
        return arr2d_batch, arr2d_xy
    
    pass


#******************************************************************************
#******************************************************************************

class DataGenerator_xy2(DataGenerator):
    
    def __getitem__(self, index):
    
        arr2d_batch, sArr =  super(DataGenerator_xy2, self).__getitem__(index)

        arr2d_xy      = np.zeros(shape=(arr2d_batch.shape[0], 4))
        arr2d_xy[:,0] = sArr[:]['true_x']
        arr2d_xy[:,1] = sArr[:]['true_y']
        arr2d_xy[:,2] = sArr[:]['x']
        arr2d_xy[:,3] = sArr[:]['y']
        
        return arr2d_batch, arr2d_xy
    
    pass
