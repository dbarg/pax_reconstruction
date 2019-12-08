
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

import json

from nn_s2waveforms_base import *

from generator_waveforms import *
from keras import backend as K
from keras import optimizers
from keras import regularizers
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from keras.utils import multi_gpu_model

import tensorflow as tf
from tensorflow.python.client import device_lib

from utils_slurm import *

proc = psutil.Process(os.getpid())





#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

class nn_xy_s2waveforms(nn_waveforms):
    
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    def train(self):
        
        #----------------------------------------------------------------------
        #----------------------------------------------------------------------
        
        config                      = tf.compat.v1.ConfigProto()
        config.log_device_placement = True
        
        
        #----------------------------------------------------------------------
        # Reduce Memory
        #----------------------------------------------------------------------
        
        K.set_floatx('float16')
        K.set_epsilon(1e-2) 
                  
        
        #----------------------------------------------------------------------
        # 0 Values means Keras chooses?
        # Anything to gain here?
        #----------------------------------------------------------------------
        
        intra = tf.compat.v1.config.threading.get_intra_op_parallelism_threads()
        inter = tf.compat.v1.config.threading.get_inter_op_parallelism_threads()

        print()
        print("intra_op_parallelism_threads: {0}".format(intra))
        print("inter_op_parallelism_threads: {0}".format(inter))
        print()
        
        #config.inter_op_parallelism_threads=2
        #config.intra_op_parallelism_threads=2
        
    
        #----------------------------------------------------------------------
        #----------------------------------------------------------------------
      
        if (False):
            
            #physical_devices = tf.config.experimental.list_physical_devices('CPU')
            #assert len(physical_devices) > 0, "Not enough CPU hardware devices available"
            #for i, dev in enumerate(physical_devices):
            #    print("{0} {1}".format(i, dev))
        
            env = get_slurm_env()
            print(env)
            
            tfconfig = get_tf_config()
            print("\n\ntfconfig:")
            print(tfconfig)
            nodename = os.environ.get('SLURM_NODENAME')
            print(nodename)
            
            os.environ['TF_CONFIG'] = json.dumps(tfconfig)
            
            
        #----------------------------------------------------------------------
        # CPU
        #----------------------------------------------------------------------
        
        print("\n------- Compiling Model -------\n")
        
        if (not self.isGpu):
            
            #sess = tf.compat.v1.Session(graph=tf.get_default_graph(), config=session_conf)
            #sess = tf.compat.v1.Session(config=config)
            #tf.compat.v1.keras.backend.set_session(sess)

            self.model = kutils.dnn_regression(self.input_dim, 2, [127], doCompile=False)
            self.model.compile(loss='mse', optimizer='adam', metrics=['acc'])
          
      
        #----------------------------------------------------------------------
        # GPU
        #----------------------------------------------------------------------
        
        else:
             
            config.gpu_options.allocator_type='BFC'
            config.gpu_options.per_process_gpu_memory_fraction=0.95
            sess = tf.compat.v1.Session(config=config)
            tf.compat.v1.keras.backend.set_session(sess)
        
            physical_devices = tf.config.experimental.list_physical_devices('GPU')
            assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
            for iGPU, dev in enumerate(physical_devices):
                tf.config.experimental.set_memory_growth(physical_devices[iGPU], True)
            
            with tf.device('/cpu:0'):
                
                if (False):
                    self.model = kutils.dnn_regression(self.input_dim, 2, [127], doCompile=False)
                else:
                    
                    self.model = Sequential()
                    kernel_reg = regularizers.l2(.001)
                    self.model.add(Dense(
                        self.input_dim, input_dim=self.input_dim, activation='relu', kernel_regularizer=kernel_reg
                    ))
                    self.model.add(Dropout(0.00005))
                    self.model.add(Dense(2))
                    self.model = multi_gpu_model(self.model, gpus=2, cpu_merge=True, cpu_relocation=True)
                    self.model.compile(loss='mse', optimizer='adam', metrics=['acc'])
                
                print("\n-> Compiled Multi-GPU model") 
        
        
        #----------------------------------------------------------------------
        #----------------------------------------------------------------------
        
        print("\n------- Data Generator -------\n")
        
        ds  = self.downsample
        epb = self.events_per_batch

        datagen_train = DataGenerator_xy(self.lst_files_train, events_per_batch=epb, downsample=ds, shuffle=True)
        datagen_test  = DataGenerator_xy2(self.lst_files_test, events_per_batch=epb, downsample=ds, shuffle=False)
        

        #----------------------------------------------------------------------
        #----------------------------------------------------------------------
        
        print("\n------- Fit -------\n")
            
        self.mem = kutils.logMemory(proc)
            
        self.history = self.model.fit_generator(
            generator=datagen_train,
            epochs=self.epochs,
            #steps_per_epoch=1,
            callbacks=[self.hist, self.mem],
            verbose=1,
            shuffle=False,
            use_multiprocessing=True,
            #max_queue_size=10,
            workers=4
        )
        
        print("Done")
        
        #os.system('touch tmp.txt')
        
        print(self.model.summary())
        #print(self.history)
        
        #self.validate(datagen_test)
        
        
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
        
    pass


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

if (__name__ == "__main__"):

    nn = nn_xy_s2waveforms()
    nn.run()
    