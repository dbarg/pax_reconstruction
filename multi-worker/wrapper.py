#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

import argparse
import os
import subprocess
import sys

from tensorflow_on_slurm import tf_config_from_slurm


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def get_resources():

    cluster, my_job_name, my_task_index = tf_config_from_slurm(ps_number=1)

    ps    = cluster['ps'][0]
    wrkrs = cluster['worker']
   
    return ps, wrkrs
   

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def wrapper():

    print("go")
    
    ps, lst_wrkrs = get_resources()
    str_wrkrs     = ','.join(lst_wrkrs)
    
    print("HERE: {0}".format(str_wrkrs))
    
    cmd1 = 'python ./asyncCluster.py --job_name={0} --task_index={1} --ps_hosts={2} --worker_hosts={3}'.format(
        'ps',
        0,
        ps,
        str_wrkrs
    )
    
    os.system(cmd1)
    
    #DETACHED_PROCESS = 0x00000008
    #pid = subprocess.Popen([sys.executable, cmd], creationflags=DETACHED_PROCESS).pid
    #pid = subprocess.Popen(
    #    [sys.executable, "longtask.py"],
    #    stdout=subprocess.PIPE,
    #    stderr=subprocess.PIPE,
    #    stdin=subprocess.PIPE
    #)
    
    return

    
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

if   __name__ == "__main__":
    
    wrapper()
    
    exit(0)