
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

from dataGenerator_waveforms import *
from nn_waveforms import *


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

class nn_xy_s2waveforms(nn_waveforms):
    
    def train(self):

        datagen_train = DataGenerator_xy(self.lst_files_train, downsample=self.downsample)
        datagen_test  = DataGenerator_xy(self.lst_files_test , downsample=self.downsample)
        
        self.model = kutils.dnn_regression(self.input_dim, 2, [127])

        self.history = self.model.fit_generator(
            generator=datagen_train,
            validation_data=datagen_test,
            initial_epoch=0,
            epochs=1,
            #steps_per_epoch=self.n_epochs_train,
            shuffle=False,
            verbose=2,
            use_multiprocessing=False,
            #workers=1,
            callbacks=[self.hist]
        )
        
            
        print(self.model.summary())

    pass


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

if (__name__ == "__main__"):

    nn = nn_xy_s2waveforms()
    nn.run()
    