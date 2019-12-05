#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

import numpy as np
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
    
    #$SLURM_JOB_ID	Job ID	5741192	$PBS_JOBID
    #$SLURM_JOBID	Deprecated. Same as SLURM_JOB_ID	 	 
    #$SLURM_JOB_NAME	Job Name	myjob	$PBS_JOBNAME
    #$SLURM_SUBMIT_DIR	Submit Directory	/lustre/payerle/work	$PBS_O_WORKDIR
    #$SLURM_JOB_NODELIST	Nodes assigned to job	compute-b24-[1-3,5-9],compute-b25-[1,4,8]	cat $PBS_NODEFILE
    #$SLURM_SUBMIT_HOST	Host submitted from	login-1.deepthought2.umd.edu	$PBS_O_HOST
    #$SLURM_JOB_NUM_NODES	Number of nodes allocated to job	2	$PBS_NUM_NODES
    #$SLURM_CPUS_ON_NODE	Number of cores/node	8,3	$PBS_NUM_PPN
    #$SLURM_NTASKS	Total number of cores for job???	11	$PBS_NP
    #$SLURM_NODEID	Index to node running on
    #relative to nodes assigned to job	0	$PBS_O_NODENUM
    #$PBS_O_VNODENUM	Index to core running on
    #within node	4	$SLURM_LOCALID
    #$SLURM_PROCID	Index to task relative to job	0	$PBS_O_TASKNUM - 1

    
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
    
    
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------

    if (isServer):
        
        print("Joining Server...")
        server.join()
        
    else:
        print("Working...")
        
        with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % 0, cluster=cluster)):

            y_true = tf.Variable( np.random.random(10) )
            y_pred = tf.Variable( np.random.random(10) )
            
            # Build model...
            #loss        = tf.nn.l2_loss(labels=y_true, y_pred)
            loss        = tf.reduce_mean(tf.squared_difference(y_true, y_pred))
            global_step = tf.contrib.framework.get_or_create_global_step()
            train_op    = tf.train.AdagradOptimizer(0.01).minimize(loss, global_step=global_step)

    
        #----------------------------------------------------------------------
        #----------------------------------------------------------------------
        
        # The StopAtStepHook handles stopping after running given steps.
        hooks=[tf.train.StopAtStepHook(last_step=10)]
    
        # The MonitoredTrainingSession takes care of session initialization,
        # restoring from a checkpoint, saving to a checkpoint, and closing when done
        # or an error occurs.
        
        with tf.train.MonitoredTrainingSession(
            master=server.target,
            #is_chief=(FLAGS.task_index == 0),
            is_chief=True,
            checkpoint_dir="/scratch/midway2/dbarge",
            hooks=hooks) as mon_sess:
            
              while not mon_sess.should_stop():
                    
                print("HERE")
                # Run a training step asynchronously.
                # See `tf.train.SyncReplicasOptimizer` for additional details on how to
                # perform *synchronous* training.
                # mon_sess.run handles AbortedError in case of preempted PS.
                mon_sess.run(train_op)
                
                continue
    
        print("Work Complete.")
    
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    return


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

if __name__ == '__main__':

    main()
    
    pass