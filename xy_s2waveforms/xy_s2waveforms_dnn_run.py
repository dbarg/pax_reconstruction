####################################################################################################
####################################################################################################

import os


####################################################################################################
####################################################################################################

n_timesteps   = 10
n_outputs     = 2
n_events      = 100000
layers_hidden = 1270

dir_input     = "./train_pax69/"
file_input    = dir_input + "array_train_input_events000000-199999_timesteps%04d.npy" % n_timesteps
file_truth    = dir_input + "array_train_truth_events000000-199999_timesteps%04d.npy" % n_timesteps




####################################################################################################
####################################################################################################

cmd = "python xy_s2waveforms_dnn_train.py " 
cmd += "-file_input %s "    % file_input
cmd += "-file_truth %s "    % file_truth
cmd += "-n_timesteps %d "   % n_timesteps
cmd += "-n_outputs %d "     % n_outputs
cmd += "-n_events %d "      % n_events
cmd += "-layers_hidden %s " % str(layers_hidden)

print("\n" + cmd + "\n")
os.system(cmd)