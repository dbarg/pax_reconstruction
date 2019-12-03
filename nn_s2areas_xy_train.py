
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

from generator_waveforms import *
from nn_s2waveforms_base import *

    
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

class nn_s2areas_xy(nn_waveforms):
    
    def train(self):

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

        print("Start")
        self.history  = self.model.fit_generator(
            generator=datagen_train,
            epochs=50,
            shuffle=False,
            callbacks=[self.hist],
            use_multiprocessing=False,
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
    