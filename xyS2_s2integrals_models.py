
##########################################################################################
##########################################################################################

import keras
from keras import backend as K
from keras import layers
from keras import regularizers
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import load_model
from keras.models import Sequential
from keras.utils import plot_model


##########################################################################################
##########################################################################################

def bargeModel_xyS2():

    ######################################################################################
    ######################################################################################
    
    name = 'barge_xyS2' # + '_' + datetime.datetime.now().strftime("%y%m%d%H%M")
    
    
    ######################################################################################
    ######################################################################################
    
    inputDim   = 127
    outputDim  = 3
    
    reg_scale  = 0.00100 # possibly bad
    bias_init  = 'zeros'
    bias_use   = True
    kern_reg   = regularizers.l2(reg_scale)
    activation = 'elu'
    keep_rate  = 0.00005
    

    ######################################################################################
    # Input Layer
    ######################################################################################
    
    model = Sequential()
    
    #model.add(Dense(inputDim, activation=activation, input_dim=inputDim))
    model.add(Dense(inputDim, input_dim=inputDim))
    
    
    ######################################################################################
    # Hidden Layer 1
    ######################################################################################
    
    model.add(Dense(
        100,
        activation         = activation,
        bias_initializer   = bias_init,
        use_bias           = bias_use,
        kernel_regularizer = kern_reg
    ))
    
    model.add(Dropout(keep_rate))
    
    
    ######################################################################################
    # Hidden Layer 2
    ######################################################################################
    
    model.add(Dense(
        80,
        activation         = activation,
        bias_initializer   = bias_init,
        use_bias           = bias_use,
        kernel_regularizer = kern_reg
    ))
    
    model.add(Dropout(keep_rate))
    
    
    ######################################################################################
    # Hidden Layer 3
    ######################################################################################
    
    model.add(Dense(
        40,
        activation         = activation,
        bias_initializer   = bias_init,
        use_bias           = bias_use,
        kernel_regularizer = kern_reg
    ))
    
    model.add(Dropout(keep_rate))
    
    
    ######################################################################################
    # Hidden Layer 1
    ######################################################################################
    
    model.add(Dense(
        20,
        activation         = activation,
        bias_initializer   = bias_init,
        use_bias           = bias_use,
        kernel_regularizer = kern_reg
    ))
    
    model.add(Dropout(keep_rate))
    
   
    ######################################################################################
    # Output Layer
    ######################################################################################
    
    model.add(Dense(outputDim))
    
 
    ######################################################################################
    # Compile model
    ######################################################################################
    
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    
    ######################################################################################
    # Output Layer
    ######################################################################################
    
    return model, name
