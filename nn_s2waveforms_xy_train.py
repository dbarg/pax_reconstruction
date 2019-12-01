
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

from generator_waveforms import *
from nn_s2waveforms_base import *

    
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

class nn_xy_s2waveforms(nn_waveforms):
    
    def train(self):

        ds  = self.downsample
        epb = self.events_per_batch
        
        datagen_train = DataGenerator_xy(self.lst_files_train, events_per_batch=epb, downsample=ds, shuffle=True)
        datagen_test  = DataGenerator_xy2(self.lst_files_test, events_per_batch=epb, downsample=ds, shuffle=False)
        
        self.model    = kutils.dnn_regression(self.input_dim, 2, [127])
        
        print("Start")
        self.history  = self.model.fit_generator(
            generator=datagen_train,
            epochs=1,
            #steps_per_epoch=1,
            callbacks=[self.hist],
            verbose=2,
            shuffle=False,
            use_multiprocessing=False
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
    