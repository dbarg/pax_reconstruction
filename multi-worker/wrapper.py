#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

import os
import re

import tensorflow as tf

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#def get_nodes(nodeliststr):
#
#    prefix, ids = re.findall("(.*-)(.*)", nodeliststr)[0]
#    nodeliststr = nodeliststr.replace(']', '').replace('[', '')
#    pre, start, stop = nodeliststr.split('-')
#    i0     = int(start)
#    i1     = int(stop)
#    server = pre + '-' + start + dom 
#    wrkrs  = list()
#    srvrs  = list()
#    srvrs.append(server)
#    
#    for i in range(i0+1, i1+1):
#        w = pre + "-{0:04d}".format(i) + dom
#        wrkrs.append(w)
#        continue
#    
#    return srvrs, wrkrs


#srvrs, wrkrs = get_nodes(nodelist)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def main():
    
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    nodename  = os.environ.get('SLURMD_NODENAME')
    nodelist  = os.environ.get('SLURM_JOB_NODELIST')
    numnodes  = os.environ.get('SLURM_JOB_NUM_NODES')
    pattern   = re.match(r"(.*-)\[(.*)-(.*)\]", nodelist)
    pre       = pattern.group(1)
    i0        = pattern.group(2)
    i1        = pattern.group(3)
    job_name  = None
    server    = pre + i0 
    worker    = pre + i1
    isServer  = (nodename == server)
    
    
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------

    print()
    print("\n-------")
    
    if (isServer):
        print("\n -> Server")
        job_name='ps'
    else:
        print("\n -> Worker")
        job_name='worker'
    
    #os.system('hostname')
    print(nodename)
    print(nodelist)
    print("servers: {0}".format(server))
    print("workers: {0}".format(worker))


    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------

    port    = '2222'
    cluster = tf.train.ClusterSpec(
        {
            "ps":     [server + ':' + port],
            "worker": [worker + ':' + port]
        }
    )
    
    server = tf.train.Server(
        cluster,
        job_name=job_name,
        task_index=0
    )
    
    if (isServer):
        print("Joining Server...")
        #server.join()
    else:
        print("HERE")

    
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    return


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

if __name__ == '__main__':

    main()
    
    pass