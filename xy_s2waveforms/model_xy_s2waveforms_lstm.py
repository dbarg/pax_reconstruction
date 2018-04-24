
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
    keep_rate=0.00005,
    go_backwards=False,
    unroll=False):

    #return_state=False,
    #stateful=False,

    ######################################################################################
    ######################################################################################
    
    name = 'model_xy_s2waveforms_dnn_' + activation

    reg_scale  = 0.00100 # possibly bad
    bias_init  = 'zeros'
    bias_use   = True
    kern_reg   = regularizers.l2(reg_scale)
    activation = 'elu'
    keep_rate  = 0.00005
    
    
    ####################################################################################################
    ####################################################################################################

    inputDim = n_timesteps*n_channels
    
    model = Sequential()
    model.add(Dense(inputDim, input_dim=inputDim))

    
    ######################################################################################
    # Hidden Layer 1
    ######################################################################################
    
    model.add(Dense(
        inputDim,
        activation         = activation,
        bias_initializer   = bias_init,
        use_bias           = bias_use,
        kernel_regularizer = kern_reg
    ))
    
    model.add(Dropout(keep_rate))
    model.add(Dense(n_outputs, activation=activation))
    
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    
    ######################################################################################
    # Save Model
    ######################################################################################
    
    folder   = "models/"    
    name_h5  = folder + name + ".h5"
    name_png = folder + name + ".png"
    
    plot_model(model, to_file=name_png, show_layer_names=True, show_shapes=True)
    model.save(name_h5, overwrite=True)
    
    
    ######################################################################################
    ######################################################################################
    
    return model, name



##########################################################################################
##########################################################################################

def lstmModel(
    n_channels,
    n_timesteps,
    n_outputs,
    #activation='sigmoid',
    activation='elu',
    keep_rate=0.00005,
    go_backwards=False,
    unroll=False):

    #return_state=False,
    #stateful=False,

    ######################################################################################
    ######################################################################################
    
    name = 'model_xy_s2waveforms_lstm_' + activation


    ####################################################################################################
    # To Do: understand stateful, return_state
    ####################################################################################################

    lstm = LSTM(
        n_channels,
        input_shape=(n_timesteps, n_channels),
        return_sequences=True,
        go_backwards=go_backwards,
        unroll=unroll
    )
    
    model = Sequential()
    model.add(lstm)
    model.add(LSTM(n_channels))
    model.add(Dropout(keep_rate))
    model.add(Dense(n_outputs, activation=activation))
    
    #model.compile(
    #    loss='binary_crossentropy',
    #    optimizer='rmsprop'#,
    #    #metrics=['accuracy']
    #)
    
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    
    ######################################################################################
    # Save Model
    ######################################################################################
    
    folder   = "models/"    
    name_h5  = folder + name + ".h5"
    name_png = folder + name + ".png"
    
    plot_model(model, to_file=name_png, show_layer_names=True, show_shapes=True)
    model.save(name_h5, overwrite=True)
    
    
    ######################################################################################
    ######################################################################################
    
    #print()
    #display(Image.open(name_png))
    #print()
        
    
    ######################################################################################
    ######################################################################################
    
    return model, name


