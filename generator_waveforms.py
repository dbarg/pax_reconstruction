
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

import inspect
import numpy as np
import keras
import os

import utils_keras as kutils
import utils_kernel as kernutils

from utils_python import *

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
        shuffle=False,
        verbose=True
    ):
        np.random.seed(13)
        
        #----------------------------------------------------------------------
        #----------------------------------------------------------------------

        self.lst_files        = lst_files
        self.shuffle          = shuffle
        self.n_files          = len(self.lst_files)
        self.events_per_file  = events_per_file
        self.events_per_batch = events_per_batch
        self.n_events         = int(self.n_files*self.events_per_file)
        self.n_batches        = int(self.n_events / self.events_per_batch)
        self.n_inputs         = n_inputs
        self.n_outputs        = n_outputs
        self.batch            = 0
        self.downsample       = downsample
        self.file_indexes     = np.arange(0, self.events_per_file, 1)
        
        
        assert(self.events_per_batch <= self.events_per_file)
        
        
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
        
        self.on_epoch_end()
        
        
        #----------------------------------------------------------------------
        #----------------------------------------------------------------------
        
        return

    
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    def __getitem__(self, index):

        #------------------------------------------------------------------------------
        #------------------------------------------------------------------------------

        sArr  = None
        iFile = int(index*self.events_per_batch / self.events_per_file)
        fpath = self.lst_files[iFile]
        i0    = int((index*self.events_per_batch)%self.events_per_file)
        i1    = int(i0 + self.events_per_batch)
        j0    = i0 + iFile*self.events_per_file
        j1    = j0 + self.events_per_batch
        
        with np.load(fpath) as data:
            sArr = data['arr_0']
                 
        batch_indices = self.file_indexes [i0:i1] # shuffled at epoch end
        #print("\n   batch indices: {0}\n".format(batch_indices))
        
        #sArr = sArr[i0:i1][:][:]
        sArr          = sArr[batch_indices][:][:]
        arr3d         = sArr[:][:]['image']
        arr3d_ds      = kernutils.downsample_arr3d(arr3d, self.downsample)
        arr3d_batch   = arr3d_ds[:][:] 
        arr2d_batch   = arr3d_batch.reshape(arr3d_batch.shape[0], arr3d_batch.shape[1]*arr3d_batch.shape[2])



        #------------------------------------------------------------------------------
        #------------------------------------------------------------------------------

        str_idxs = ""
        
        if (not self.shuffle):
            str_idxs = ", i0={0:04d}, i1={1:04d}, j0={2:05d}, j1={3:05d}".format(i0, i1, j0, j1)
        
        print("   (Data Generator) Batch: {0}/{1}, File {2}: {3}, Memory: {4} GB{5}".format(
            index,
            self.n_batches,
            iFile,
            os.path.basename(fpath),
            getMemoryGB(proc),
            str_idxs
        ))


        #------------------------------------------------------------------------------
        #------------------------------------------------------------------------------
        
        assert not np.any(np.isnan(arr2d_batch))
        
        return arr2d_batch, sArr
        

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    def on_epoch_end(self):

        #time.sleep(5) # Sometimes tensorflow print output lags
        #print(__name__ + "." + inspect.currentframe().f_code.co_name + "()\n")
        #print()
        
        if (self.shuffle):
            print("Shuffling...")
            np.random.shuffle(self.file_indexes)
            
        return
    

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    def __len__(self):
        #return 2
        #print(__name__ + "." + inspect.currentframe().f_code.co_name + "()")
        return self.n_batches # Batches per dataset/epoch

    
#******************************************************************************
#******************************************************************************

class DataGenerator_s2areas_xy(DataGenerator):
    
    def __getitem__(self, index):
    
        arr2d_batch, sArr = super(DataGenerator_s2areas_xy, self).__getitem__(index)

        arr2d_xy       = np.zeros(shape=(arr2d_batch.shape[0], 2))
        arr2d_xy[:,0]  = sArr[:]['true_x']
        arr2d_xy[:,1]  = sArr[:]['true_y']
        arr_s2areas    = sArr[:]['s2_areas']
        
        s2area_sums    = np.sum(arr_s2areas, axis=1)
        s2area_sum_min = np.amin(s2area_sums)
        
        assert(s2area_sum_min > 0)
        
        return arr_s2areas, arr2d_xy

    
#******************************************************************************
#******************************************************************************

class DataGenerator_s2areas_xy2(DataGenerator):
    
    def __getitem__(self, index):
    
        arr2d_batch, sArr =  super(DataGenerator_s2areas_xy2, self).__getitem__(index)

        arr_s2areas   = sArr[:]['s2_areas']
        arr2d_xy      = np.zeros(shape=(arr2d_batch.shape[0], 4))
        arr2d_xy[:,0] = sArr[:]['true_x']
        arr2d_xy[:,1] = sArr[:]['true_y']
        arr2d_xy[:,2] = sArr[:]['x']
        arr2d_xy[:,3] = sArr[:]['y']
        
        return arr_s2areas, arr2d_xy

    
#******************************************************************************
#******************************************************************************

class DataGenerator_xy(DataGenerator):
    
    def __getitem__(self, index):
    
        arr2d_batch, sArr =  super(DataGenerator_xy, self).__getitem__(index)
        
        arr2d_xy      = np.zeros(shape=(arr2d_batch.shape[0], 2))
        arr2d_xy[:,0] = sArr[:]['true_x']
        arr2d_xy[:,1] = sArr[:]['true_y']
        
        return arr2d_batch, arr2d_xy


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
