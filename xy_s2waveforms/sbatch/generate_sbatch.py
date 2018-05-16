
####################################################################################################
####################################################################################################

import glob
import os
import re
import stat
import sys


####################################################################################################
####################################################################################################

#n_timesteps   = 10
#layers_hidden = [1270]
#layers_hidden = [1270, 127]

n_timesteps   = 20
layers_hidden = [2540]
#layers_hidden = [2540, 1270, 127]

#n_timesteps   = 50
#layers_hidden = [6350]
#layers_hidden = [6350, 2540, 1270, 127]


####################################################################################################
####################################################################################################

n_channels    = 127
n_inputs      = n_timesteps * n_channels
n_outputs     = 2
n_events      = 100000
n_epochs      = 10

layers_arg    = str(' ').join(str(x) for x in layers_hidden)
layers_desc   = 'layers' + str(n_inputs) + '-' + str('-').join(str(x) for x in layers_hidden) + '-' + str(n_outputs)
desc          = 'timesteps%04d' % n_timesteps + '_' + layers_desc 

dir_input     = "../train_pax65/"
file_input    = dir_input + "array_train_input_events000000-199999_timesteps%04d.npy" % n_timesteps
file_truth    = dir_input + "array_train_truth_events000000-199999_timesteps%04d.npy" % n_timesteps

line_cmd = "python ../xy_s2waveforms_dnn_train.py " 
line_cmd += "-file_input %s "    % file_input
line_cmd += "-file_truth %s "    % file_truth
line_cmd += "-n_timesteps %d "   % n_timesteps
line_cmd += "-n_outputs %d "     % n_outputs
line_cmd += "-n_events %d "      % n_events
line_cmd += "-n_epochs %d "      % n_epochs
line_cmd += "-layers_hidden %s " % layers_arg


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

useGPU = True

line_python = 'source ~/.setup-ml_py364.sh'
line_sbatch = (
               "#!/bin/bash\n\n"
               "#SBATCH --output=%s/logs/" % dir_output + 'out_%s' % desc + '.txt' + "\n"
               "#SBATCH --error=%s/logs/"  % dir_output + 'err_%s' % desc + '.txt' + "\n"
               "#SBATCH --ntasks=1\n"
               "#SBATCH --account=pi-lgrandi\n\n"
              )

if (useGPU is True):

    line_python = 'source ~/.setup-ml_gpu2.sh'
    line_sbatch += (
                    "#SBATCH --partition=gpu2\n"
                    "#SBATCH --gres=gpu:1\n"
                    "#SBATCH --nodes=1\n"
                    "#SBATCH --ntasks=1\n"
                    "#SBATCH --time=01:00:00\n"
                   )

    line_gpu = (
            'echo ""\n'
            'nvidia-smi\n'
            'echo ""\n'
            'nvcc --version\n'
            'echo ""\n\n'
           )



####################################################################################################
####################################################################################################

base_file  = desc
batch_file = dir_scripts + 'submit_' + base_file + ".sh"

with open(batch_file, 'w') as out_file:

    out_file.write(line_sbatch)
    out_file.write('\n')
    if (useGPU): out_file.write(line_gpu)
    out_file.write(line_python)
    out_file.write('\n\n')
    out_file.write(line_cmd)
    
os.chmod(batch_file, 0o744)

print(batch_file)


