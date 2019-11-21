
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

import ipca_helpers
import utils_keras as kutils

import keras
#from keras.callbacks import Callback
#import keras.callbacks.Callback

import utils_keras as kutils

proc = psutil.Process(os.getpid())


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def getMemoryGB(proc):
    
    mem_gb = "{0:.1f}".format(proc.memory_info().rss/1e9)
    
    return mem_gb 


#**********************************************************************************
#**********************************************************************************

class nn_xy_s2waveforms():
    
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
    
    def __init__(self):
        
        #------------------------------------------------------------------------------
        #------------------------------------------------------------------------------

        self.maxRows          = 200
        self.args             = parse_arguments()
        self.events_per_batch = self.args.events_per_batch 
        self.downsample       = self.args.downsample
        dir_data              = self.args.directory
        self.n_batches        = int(self.maxRows/self.events_per_batch)
        self.input_dim        = int(127000/self.downsample)

        assert(os.path.exists(dir_data))

        self.max_dirs         = min(self.args.max_dirs, 9)
        self.lst_dir_files    = glob.glob(dir_data + "/strArr*.npz")
        self.lst_dir_files    = sorted(self.lst_dir_files)
        self.lst_files_train  = self.lst_dir_files[0:min(self.max_dirs,7)]
        self.lst_files_test   = self.lst_dir_files[7:8]
        n_dir                 = len(self.lst_dir_files)
        n_files_test          = len(self.lst_files_test)
        n_files_train         = len(self.lst_files_train)
        self.n_events_train   = n_files_train*self.maxRows
        
        self.n_epochs_train   = int( (self.maxRows) / (self.events_per_batch) )*n_files_train
        self.arr2d_pred       = np.zeros(shape=(self.maxRows*n_files_test, 6))

        assert(1000 % self.downsample == 0)
        
        #------------------------------------------------------------------------------
        #------------------------------------------------------------------------------
        
        print("Input Directory:  {0}".format(dir_data))
        print("Downsample:       {0}".format(self.downsample))
        print("Input dimension:  {0}".format(self.input_dim))
        print("Batches:          {0}".format(self.n_batches))
        print("Events per batch: {0}".format(self.events_per_batch))
        print("Events:           {0}".format(self.n_events_train))
        
        print("Train files: \n")
        for x in (self.lst_files_train):
            print("   " + x)
            
        print("Test files: \n")
        for x in (self.lst_files_test):
            print("   " + x)
  

        
        #----------------------------------------------------------------------
        #----------------------------------------------------------------------
    
        self.init_model()
        
        return
   
    
    
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
    
    def func_yield(self, arr2d_batch, arr2d_xy):
        
          yield(
              {'dense_1_input': arr2d_batch},
              {'dense_3'      : arr2d_xy}
          )
        
    
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
    
    def generator_waveform_xy(self, x):
        
        print("generator_waveform_xy")
        
        while(True):
            
            print("while")
            
            #------------------------------------------------------------------------------
            #------------------------------------------------------------------------------
    
            for iFile, fpath in enumerate(self.lst_files_train):
        
                #----------------------------------------------------------------------
                #----------------------------------------------------------------------

                print("\n   Loading data file: {0} ...".format(fpath))

                #sArr     = np.load(fpath)
                sArr     = np.load(fpath)['arr_0']
                sArr     = sArr[0:self.maxRows][:][:]
                arr3d    = sArr[:][:]['image']
                arr3d_ds = ipca_helpers.downsample_arr3d(arr3d, self.downsample)
                j0       = iFile*self.maxRows
                j1       = j0 + self.maxRows - 1


                #----------------------------------------------------------------------
                #----------------------------------------------------------------------

                for ibatch in range(0, self.n_batches):

                    i0  = int(ibatch*self.events_per_batch)
                    i1  = int(i0 + self.events_per_batch - 1)
 
                    arr3d_batch    = arr3d_ds[i0:i1+1][:] 
                    arr2d_batch    = arr3d_batch.reshape(arr3d_batch.shape[0], arr3d_batch.shape[1]*arr3d_batch.shape[2])
                    arr2d_xy       = np.zeros(shape=(arr2d_batch.shape[0], 2))
                    arr2d_xy[:, 0] = sArr[i0:i1+1][:]['true_x']
                    arr2d_xy[:, 1] = sArr[i0:i1+1][:]['true_y']
        
                    print("   NN Fitting batch {0}/{1}   (events: {2}-{3})   (Memory Usage: {4} GB)".format(
                        ibatch+1,
                        self.n_batches,
                        j0,
                        j1,
                        getMemoryGB(proc)
                    ))
        
                    #self.func_yield(arr2d_batch, arr2d_xy)
        
                    yield(
                        {'dense_1_input': arr2d_batch},
                        {'dense_3'      : arr2d_xy}
                    )


                #----------------------------------------------------------------------
                #----------------------------------------------------------------------

                continue
    
            continue
        
        return
        

    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
    
    def validate_model(self):
        
        print("\n--- Test ---")
        
        #------------------------------------------------------------------------------
        #------------------------------------------------------------------------------
    
        for iFile, fpath in enumerate(self.lst_files_test):
    
            #----------------------------------------------------------------------
            #----------------------------------------------------------------------

            print("\n   Loading data file: {0} ...".format(fpath))

            sArr     = np.load(fpath)
            sArr     = sArr[0:self.maxRows][:][:]
            arr3d    = sArr[:][:]['image']
            arr3d_ds = ipca_helpers.downsample_arr3d(arr3d, self.downsample)
            j0       = iFile*self.maxRows
            j1       = j0 + self.maxRows - 1


            #----------------------------------------------------------------------
            #----------------------------------------------------------------------

            for ibatch in range(0, self.n_batches):

                i0  = int(ibatch*self.events_per_batch)
                i1  = int(i0 + self.events_per_batch)
 
                arr3d_batch       = arr3d_ds[i0:i1][:] 
                arr2d_batch       = arr3d_batch.reshape(arr3d_batch.shape[0], arr3d_batch.shape[1]*arr3d_batch.shape[2])
                
                arr2d_xy          = np.zeros(shape=(arr2d_batch.shape[0], 2))
                arr2d_xy_s2       = np.zeros(shape=(arr2d_batch.shape[0], 2))
                arr2d_xy[:, 0]    = sArr[i0:i1][:]['x_true']
                arr2d_xy[:, 1]    = sArr[i0:i1][:]['y_true']
                arr2d_xy_s2[:, 0] = sArr[i0:i1][:]['x_s2']
                arr2d_xy_s2[:, 1] = sArr[i0:i1][:]['y_s2']
                
                print("\n   Evaluating Input Data: Predict batch {0}/{1}   (events: {2}-{3})".format(
                    ibatch+1,
                    self.n_batches,
                    i0,
                    i1
                ))
                
                print("    -> Memory Usage: {0} GB".format(getMemoryGB(proc)))
                
                print("Predict...")
                
                arr2d_xy_pred = self.model.predict(arr2d_batch)
        
                self.arr2d_pred[i0:i1, 0] = arr2d_xy[:,0]
                self.arr2d_pred[i0:i1, 1] = arr2d_xy[:,1]
                self.arr2d_pred[i0:i1, 2] = arr2d_xy_s2[:,0]
                self.arr2d_pred[i0:i1, 3] = arr2d_xy_s2[:,1]
                self.arr2d_pred[i0:i1, 4] = arr2d_xy_pred[:,0]
                self.arr2d_pred[i0:i1, 5] = arr2d_xy_pred[:,1]
            
                #for i in range(0, 100):
                #    print()
                #    print("true (x,y) = ({0}, {1})".format(self.arr2d_pred[i, 0], self.arr2d_pred[i, 1]))
                #    print("reco (x,y) = ({0}, {1})".format(self.arr2d_pred[i, 2], self.arr2d_pred[i, 3]))
                #    print("pred (x,y) = ({0}, {1})".format(self.arr2d_pred[i, 4], self.arr2d_pred[i, 5]))
                      
                continue
            

            #----------------------------------------------------------------------
            #----------------------------------------------------------------------

            continue
    
        return
        

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    def init_model(self):
        
        self.model = kutils.dnn_regression(self.input_dim, 2, [127])
        self.hist  = kutils.logHistory()
        
        return

    
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    def train(self):
        
        print("\n--- Train ---")
        
        self.history = self.model.fit_generator(
            self.generator_waveform_xy(0),
            initial_epoch=0,
            steps_per_epoch=2,#self.n_epochs_train,
            epochs=1,
            shuffle=False,
            verbose=1,
            workers=1,
            use_multiprocessing=False,
            callbacks=[self.hist]
            #validation_data=None,
            #validation_steps=None,
            #validation_freq=1,
            #class_weight=None,
            #max_queue_size=10,
        )
        
        print(self.model.summary())

        
        return

    
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    def save(self):
    
        #----------------------------------------------------------------------
        # To Do, Add:
        #   dropout rate, learning rate
        #----------------------------------------------------------------------
        
        acc         = int(100*np.round(self.history.history['acc'], 2))
        layers_desc = kutils.getModelDescription(self.model)
        f_model     = 'nn_modl_acc{0:.0f}_evts{1:05d}_{2}.h5'.format(acc, self.n_events_train, layers_desc)
        f_pred      = 'nn_pred_acc{0:.0f}_evts{1:05d}_{2}.h5'.format(acc, self.n_events_train, layers_desc)
        f_hist      = 'nn_hist_acc{0:.0f}_evts{1:05d}_{2}.h5'.format(acc, self.n_events_train, layers_desc)

        
        #----------------------------------------------------------------------
        # Save
        #----------------------------------------------------------------------
        
        print()
        print("Saving '{0}'...".format(f_model))
        print("Saving '{0}'...".format(f_pred))
              
        self.model.save(f_model)                                # Model
        np.save(f_pred, self.arr2d_pred)                        # Predictions
        arrHist      = np.zeros(shape=(len(self.hist.accs), 3)) # History
        arrHist[:,0] = np.array(self.hist.losses)
        arrHist[:,1] = np.array(self.hist.accs)
        arrHist[:,2] = np.array(self.hist.times)
        np.save(f_hist, arrHist)   

        assert(self.hist.batches == arrHist.shape[0])
        
        # What about multiple Epochs?
        print()
        print("Batches:          {0}".format(self.hist.batches))
        print("Events per Batch: {0}".format(self.events_per_batch))
        print("Epochs:           {0}".format(self.n_epochs_train))
        print("Events:           {0}".format(self.n_events_train))

        assert(self.hist.batches*self.events_per_batch == self.n_events_train)
        
        
        #----------------------------------------------------------------------
        #----------------------------------------------------------------------
        
        return
    
    
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('-max_dirs'        , required=True, type=int)
    parser.add_argument('-directory'       , required=True)
    parser.add_argument('-events_per_batch', required=True, type=int)
    parser.add_argument('-samples'         , required=True, type=int)
    parser.add_argument('-downsample'      , required=True, type=int)
    
    arguments = parser.parse_args()
    
    return arguments


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

if (__name__ == "__main__"):

    print("\nStarting...\n")
    
    t1 = time.time()
    
    nn = nn_xy_s2waveforms()
    nn.train()
    #nn.validate_model()
    nn.save()
    
    t2 = time.time()
    dt = (t2 - t1)/60
    
    print("\nDone in {0:.1f} min".format(dt))
    

