
####################################################################################################
####################################################################################################

import glob
import os
import re
import stat
import sys


####################################################################################################
####################################################################################################

layers_hidden = [1270, 1270]
n_timesteps   = 10


####################################################################################################
####################################################################################################

n_channels    = 127
n_inputs      = n_timesteps * n_channels
n_outputs     = 2
n_events      = 100000

layers_arg    = str(' ').join(str(x) for x in layers_hidden)
layers_desc   = 'layers' + str(n_inputs) + '-' + str('-').join(str(x) for x in layers_hidden) + '-' + str(n_outputs)
desc          = 'timesteps%04d' % n_timesteps + '_' + layers_desc 

dir_input     = "../train_pax65/"
file_input    = dir_input + "array_train_input_events000000-199999_timesteps%04d.npy" % n_timesteps
file_truth    = dir_input + "array_train_truth_events000000-199999_timesteps%04d.npy" % n_timesteps

cmd = "python ../xy_s2waveforms_dnn_train.py " 
cmd += "-file_input %s "    % file_input
cmd += "-file_truth %s "    % file_truth
cmd += "-n_timesteps %d "   % n_timesteps
cmd += "-n_outputs %d "     % n_outputs
cmd += "-n_events %d "      % n_events
cmd += "-layers_hidden %s " % layers_arg


####################################################################################################
####################################################################################################

dir_output  = '/home/dbarge/reconstruction/xy_s2waveforms/sbatch/'
dir_logs    = dir_output + '/logs'
dir_scripts = './scripts/'


####################################################################################################
####################################################################################################

if (not os.path.isdir(dir_logs)):
    
    os.mkdir(dir_logs)
    print("Created '" + dir_logs + "'")


####################################################################################################
####################################################################################################

line0 = '#!/bin/bash'
line1 = '#SBATCH --output=%s/logs/' % dir_output + 'out_%s' % desc + '.txt'
line2 = '#SBATCH --error=%s/logs/'  % dir_output + 'err_%s' % desc + '.txt'
line3 = '#SBATCH --ntasks=1'
line4 = '#SBATCH --account=pi-lgrandi'
line5 = 'source /home/dbarge/.bashrc'
line6 = 'source /home/dbarge/.setup-ml_py364.sh'
line7 = 'echo "\n$PATH\n"'
line8 = cmd


####################################################################################################
####################################################################################################

base_file  = desc
batch_file = dir_scripts + 'submit_' + base_file + ".sh"

with open(batch_file, 'w') as out_file:

    out_file.write(line0)
    out_file.write('\n\n')
    out_file.write(line1)
    out_file.write('\n')
    out_file.write(line2)
    out_file.write('\n')
    out_file.write(line3)
    out_file.write('\n')
    out_file.write(line4)
    out_file.write('\n\n')
    out_file.write(line5)
    out_file.write('\n')
    out_file.write(line6)
    out_file.write('\n\n')
    out_file.write(line7)
    out_file.write('\n\n')
    out_file.write(line8)
    out_file.write('\n')
    
os.chmod(batch_file, 0o744)

print(batch_file)


