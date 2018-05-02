
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

def xy_s2integrals_dnnModel(activation='elu', keep_rate=0.00005):

    ######################################################################################
    ######################################################################################
    
    name = 'xy_s2integrals_dnnModel_' + activation
    
    
    ######################################################################################
    ######################################################################################
    
    inputDim   = 127
    outputDim  = 2
    
    reg_scale  = 0.00100 # possibly bad
    bias_init  = 'zeros'
    bias_use   = True
    kern_reg   = regularizers.l2(reg_scale)
    

    ######################################################################################
    # Input Layer
    ######################################################################################
    
    model = Sequential()
    model.add(Dense(inputDim, input_dim=inputDim))
    
    
    ######################################################################################
    # Hidden Layer 1
    ######################################################################################
    
    model.add(Dense(
        127,
        activation         = activation,
        bias_initializer   = bias_init,
        use_bias           = bias_use,
        kernel_regularizer = kern_reg
    ))
    
    
    ######################################################################################
    ######################################################################################
    
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


##########################################################################################
##########################################################################################

def bargeModel():

    ######################################################################################
    ######################################################################################
    
    name = 'barge_xy' # + '_' + datetime.datetime.now().strftime("%y%m%d%H%M")
    
    
    ######################################################################################
    ######################################################################################
    
    inputDim   = 127
    outputDim  = 2
    
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
    # Hidden Layer 4
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


##########################################################################################
# From Xenon1T
##########################################################################################

def Xenon1TPosRecMLP(
    inputDim = 127,
    activation='elu', 
    reg_scale=0.00100, # Regularization on coefficients
    keep_rate=0.00005, # The smaller the keep rate, the better the performance is
    noutput=2):
        

    name = 'xenon1t'
    
    ######################################################################################
    # hidden layers to use, for reference, Yuehuan used [32,28]
    ######################################################################################
    
    hidden_layers = [100, 80, 40, 20]
    

    ######################################################################################
    # Input Layer
    ######################################################################################
    
    model = Sequential() 
    model.add(Dense(inputDim, input_dim=inputDim))

    
    ######################################################################################
    # Hidden Layers
    ######################################################################################
    
    for hidden_layer in hidden_layers:
        
        model.add(Dense(
            hidden_layer,
            kernel_initializer='normal',
            activation=activation,
            bias_initializer='zeros',
            use_bias=True,
            kernel_regularizer = regularizers.l2(reg_scale)
        ))
        
        model.add(Dropout(keep_rate))
        
        continue
        
        
    ######################################################################################
    # Output Layer
    ######################################################################################
    
    model.add(Dense(noutput)) # Output layer
    
    
    ######################################################################################
    # Compile model
    ######################################################################################
    
    model.compile(loss='mean_squared_error', optimizer='adam')

    
    ######################################################################################
    ######################################################################################
    
    return model, name




##########################################################################################
##########################################################################################

#model.compile(loss='mean_absolute_error'            , optimizer='adam', metrics=['accuracy'])
#model.compile(loss='mean_absolute_percentage_error' , optimizer='adam', metrics=['accuracy'])
#model.compile(loss='mean_squared_logarithmic_error' , optimizer='adam', metrics=['accuracy'])
#model.compile(loss='squared_hinge'                  , optimizer='adam', metrics=['accuracy'])
#model.compile(loss='hinge'                          , optimizer='adam', metrics=['accuracy'])
#model.compile(loss='categorical_hinge'              , optimizer='adam', metrics=['accuracy'])
#model.compile(loss='logcosh'                        , optimizer='adam', metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy'       , optimizer='adam', metrics=['accuracy'])
#model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.compile(loss='binary_crossentropy'            , optimizer='adam', metrics=['accuracy'])
#model.compile(loss='kullback_leibler_divergence'    , optimizer='adam', metrics=['accuracy'])
#model.compile(loss='poisson'                        , optimizer='adam', metrics=['accuracy'])
#model.compile(loss='cosine_proximity'               , optimizer='adam', metrics=['accuracy'])
    
    