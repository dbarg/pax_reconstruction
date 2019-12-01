
####################################################################################################
####################################################################################################

#from utils_python.python_imports import *
#from keras_utils.keras_imports   import *

import utils_keras as kutils


####################################################################################################
####################################################################################################

useGPU = False

dir_input  = '/project/lgrandi/dbarge/reconstruction/xy_s2waveforms/train_683/'
dir_output = '/project/lgrandi/dbarge/reconstruction/xy_s2waveforms/sbatch/'


####################################################################################################
####################################################################################################

#n_timesteps   = 10
#n_timesteps   = 19
#n_timesteps   = 25
n_timesteps   = 38

layers_hidden = [1270, 127]

n_channels    = 127
n_inputs      = n_timesteps * n_channels
n_outputs     = 2
n_events      = 100000
n_epochs      = 10

file_input    = dir_input + "array_train_input_events000000-199999_timesteps%04d.npy" % n_timesteps
file_truth    = dir_input + "array_train_truth_events000000-199999_timesteps%04d.npy" % n_timesteps


####################################################################################################
####################################################################################################

def getCommand(
    file_input,
    file_truth,
    n_timesteps,
    n_outputs,
    n_events,
    n_epochs,
    layers_hidden,
    loss='mean_squared_error',
    optimizer='adam',
    useGPU=False):
    
    ####################################################################################################
    ####################################################################################################
    
    loss_desc = kutils.getLossDescription(loss)
    layr_desc = kutils.getLayerDescription(layers_hidden)
    layers_desc   = 'layers' + str(n_inputs) + '-' + layr_desc + '-' + str(n_outputs)

    desc          =  ('ts%04d' % n_timesteps) + '_' + ('e%02d' % n_epochs) + '_' + loss_desc + '_' + optimizer + '_' + layers_desc 
    
    
    ####################################################################################################
    ####################################################################################################
    
    cmd = "python ../xy_s2waveforms_dnn_train.py \\" 
    cmd += "\n\t-file_input %s \\"    % file_input
    cmd += "\n\t-file_truth %s \\"    % file_truth
    cmd += "\n\t-n_timesteps %d \\"   % n_timesteps
    cmd += "\n\t-n_outputs %d \\"     % n_outputs
    cmd += "\n\t-n_events %d \\"      % n_events
    cmd += "\n\t-n_epochs %d \\"      % n_epochs
    cmd += "\n\t-layers_hidden %s \\" % str(' ').join(str(x) for x in layers_hidden)
    cmd += "\n\t-loss %s \\"          % loss
    cmd += "\n\t-optimizer %s \\"     % optimizer
    cmd += "\n\t-useGPU %d \\"        % useGPU

    return cmd, desc

    

###########################################################################
# Inputs:
#  - output file
#  - command
#  - GPU
###########################################################################

def main():
    
    cmd, desc = getCommand(
        file_input,
        file_truth,
        n_timesteps,
        n_outputs,
        n_events,
        n_epochs,
        layers_hidden)
    
    kutils.writeSBatchScript(dir_output, desc, cmd, useGPU)
    


###########################################################################
###########################################################################

if __name__== "__main__":
    
    main()


