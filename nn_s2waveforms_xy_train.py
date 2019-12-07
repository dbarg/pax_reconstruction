
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

from nn_s2waveforms_base import *

from generator_waveforms import *
from keras import backend as K
from keras.utils import multi_gpu_model

import tensorflow as tf
from tensorflow.python.client import device_lib


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

class nn_xy_s2waveforms(nn_waveforms):
    
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    def train(self):
              
        #----------------------------------------------------------------------
        # Reduce Memory
        #----------------------------------------------------------------------
        
        K.set_floatx('float16')
        K.set_epsilon(1e-4) 
            
    
        #----------------------------------------------------------------------
        #----------------------------------------------------------------------

        self.parallel_model = None
      
        if (not self.isGpu):
            
            self.model = kutils.dnn_regression(self.input_dim, 2, [127], doCompile=True)
            
        else:
             
            #tf.config.gpu.set_per_process_memory_fraction(0.99)
            #tf.config.gpu.set_per_process_memory_growth(True)
            #tf.config.gpu.allocator_type('BFC')
    
            config=tf.compat.v1.ConfigProto(log_device_placement=True)
            config.gpu_options.allocator_type='BFC'
            config.gpu_options.per_process_gpu_memory_fraction=0.95
            sess = tf.compat.v1.Session(config=config)
            #tf.compat.v1.set_session(sess)  
        
            physical_devices = tf.config.experimental.list_physical_devices('GPU')
            assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
            #for iGPU, dev in enumerate(physical_devices):
            #    tf.config.experimental.set_memory_growth(physical_devices[iGPU], True)

            
            with tf.device('/cpu:0'):
                
                self.model          = kutils.dnn_regression(self.input_dim, 2, [127], doCompile=False)
                self.parallel_model = multi_gpu_model(self.model, gpus=2, cpu_merge=True, cpu_relocation=True)
                self.parallel_model.compile(loss='mse', optimizer='adam', metrics=['acc'])
                print("\n-> Compiled Multi-GPU model") 
        
        
        #----------------------------------------------------------------------
        #----------------------------------------------------------------------
        
        ds  = self.downsample
        epb = self.events_per_batch
        
        datagen_train = DataGenerator_xy(self.lst_files_train, events_per_batch=epb, downsample=ds, shuffle=True)
        datagen_test  = DataGenerator_xy2(self.lst_files_test, events_per_batch=epb, downsample=ds, shuffle=False)
        
      
        print("\n------- Fit Generator -------\n")
        #self.history = self.model.fit_generator(
        self.history = self.parallel_model.fit_generator(
            generator=datagen_train,
            epochs=1,
            #steps_per_epoch=1,
            callbacks=[self.hist],
            verbose=1,
            shuffle=False,
            use_multiprocessing=False#,
            #max_queue_size=10,
            #workers=4
        )
        print("Done")
        
        print(self.model.summary())
        #print(self.history)
        
        self.validate(datagen_test)
        
        
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
        
    pass


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

if (__name__ == "__main__"):

    nn = nn_xy_s2waveforms()
    nn.run()
    