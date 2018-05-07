
##########################################################################################
# https://keras.io/getting-started/sequential-model-guide/
##########################################################################################

from keras import backend as K
from keras import layers
from keras import regularizers
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers.normalization import BatchNormalization
from keras.models import load_model
from keras.models import Sequential
from keras.utils import plot_model


##########################################################################################
##########################################################################################

def dnnModel(
    n_channels,
    n_timesteps,
    n_outputs,
    activation='elu',
    keep_rate=0.00005):

    ######################################################################################
    ######################################################################################

    inputDim = n_timesteps*n_channels
    
    name = 'dnn_xy_s2waveforms_' + activation #+ '_in%04d' % inputDim + '_out%01d' % n_outputs

    reg_scale  = 0.00100 # possibly bad
    bias_init  = 'zeros'
    bias_use   = True
    kern_reg   = regularizers.l2(reg_scale)
    
   
    ######################################################################################
    # Input Layer
    ######################################################################################
    
    model = Sequential()
    
    #model.add(Dense(inputDim, input_dim=inputDim))
    model.add(Dense(inputDim, input_dim=inputDim, activation=activation))

    
    ######################################################################################
    # Timesteps:
    #
    #  10
    #   layers = [635, 317]
    #
    #  50
    #   layers = [1270, 508]
    ######################################################################################
    
    #hidden_layers = [inputDim, 635, 317]
    hidden_layers = [inputDim, int(inputDim/10)]

    
    ######################################################################################
    # Hidden Layers
    ######################################################################################
    
    for layer_dim in hidden_layers:
        
        ##################################################################################
        ##################################################################################
        
        model.add(Dense(
            layer_dim,
            activation         = activation,
            bias_initializer   = bias_init,
            use_bias           = bias_use#,
            #kernel_regularizer = kern_reg
        ))
        
        model.add(Dropout(keep_rate))

        
        ##################################################################################
        ##################################################################################

        continue
        
        
    ######################################################################################
    ######################################################################################
    
    #model.add(Dense(n_outputs, activation=activation))
    model.add(Dense(n_outputs))
    
     
    ######################################################################################
    ######################################################################################
    
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['acc'])
    
    
    ######################################################################################
    ######################################################################################
    
    return model, name



##########################################################################################
##########################################################################################

def dnnModel_s2integrals(
    activation='elu',
    keep_rate=0.00005):

    ######################################################################################
    ######################################################################################
    
    name = 'model_xy_s2waveforms_dnn_s2integrals_' + activation

    reg_scale  = 0.00100 # possibly bad
    bias_init  = 'zeros'
    bias_use   = True
    kern_reg   = regularizers.l2(reg_scale)
    
    
    ####################################################################################################
    ####################################################################################################

    n_outputs = 2
    inputDim  = 127
    
    model = Sequential()
    model.add(Dense(inputDim, input_dim=inputDim))

    
   
    ######################################################################################
    # Hidden Layer
    ######################################################################################
    
    model.add(Dense(
        inputDim*4,
        activation         = activation,
        bias_initializer   = bias_init,
        use_bias           = bias_use,
        kernel_regularizer = kern_reg
    ))
    
    model.add(Dropout(keep_rate))
    
    
    ######################################################################################
    # Hidden Layer
    ######################################################################################
    
    model.add(Dense(
        inputDim,
        activation         = activation,
        bias_initializer   = bias_init,
        use_bias           = bias_use,
        kernel_regularizer = kern_reg
    ))
    
    model.add(Dropout(keep_rate))
    

    ######################################################################################
    # Hidden Layer
    ######################################################################################
    
    model.add(Dense(
        int(inputDim/2),
        activation         = activation,
        bias_initializer   = bias_init,
        use_bias           = bias_use,
        kernel_regularizer = kern_reg
    ))
    
    model.add(Dropout(keep_rate))
    
    
    ######################################################################################
    # Hidden Layer
    ######################################################################################
    
    model.add(Dense(
        int(inputDim/4),
        activation         = activation,
        bias_initializer   = bias_init,
        use_bias           = bias_use,
        kernel_regularizer = kern_reg
    ))
    
    model.add(Dropout(keep_rate))
    
     
    ######################################################################################
    ######################################################################################
    
    #model.add(Dense(n_outputs, activation=activation))
    model.add(Dense(n_outputs))
    
     
    ######################################################################################
    ######################################################################################
    
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['acc'])
    
    
    ######################################################################################
    ######################################################################################
    
    return model, name

