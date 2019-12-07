
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

import os

from keras import backend as K

#import theano
#theano.config.device = 'gpu'
#theano.config.floatX = 'float32'

#import tensorflow as tf
#from tensorflow.python.client import device_lib

from generator_waveforms import *
from nn_s2waveforms_base import *


#******************************************************************************
#******************************************************************************
    
class test(keras.callbacks.Callback):
    
    def on_epoch_end(self, epoch, logs={}):
        
        print()
        os.system('nvidia-smi')
        print()
        
    pass


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

class nn_s2areas_xy(nn_waveforms):
    
    os.environ['KERAS_BACKEND'] = 'theano'
    
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------

    def train(self, verbose=False):

        
        #----------------------------------------------------------------------
        #----------------------------------------------------------------------

        #if (verbose):
        #    tf.debugging.set_log_device_placement(True)
        
        #gpus = K.tensorflow_backend._get_available_gpus()
        #gpus  = tf.config.experimental.list_logical_devices('GPU')

        #print("Devs:           {0}".format(device_lib.list_local_devices()))
        #print("Available GPUS: {0}".format(gpus))
        #print("Available GPUS: {0}".format(gpus))
        
        
        #----------------------------------------------------------------------
        #----------------------------------------------------------------------
        
        epb = self.events_per_batch
        
        datagen_train = DataGenerator_s2areas_xy(self.lst_files_train, events_per_batch=epb, shuffle=True)
        datagen_test  = DataGenerator_s2areas_xy2(self.lst_files_test , events_per_batch=epb, shuffle=False)

        self.model = kutils.dnn_regression(
            127,
            2,
            [100, 80, 40, 20],
            dropout=0.00005,
            activation='relu'
        )

        tst = test()
        
        print("Start")
        self.history  = self.model.fit_generator(
            generator=datagen_train,
            epochs=10,
            shuffle=False,
            callbacks=[self.hist, tst],
            #use_multiprocessing=False,
            use_multiprocessing=True,
            workers=16,
            max_queue_size=20,
            verbose=2
        )
        print("Done")
        
        print(self.model.summary())
        ##print(self.history)
        
        self.validate(datagen_test)
        
        
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
        
    pass


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

if (__name__ == "__main__"):

    nn = nn_s2areas_xy()
    nn.run()
    