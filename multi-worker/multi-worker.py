
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

import keras
import tensorflow as tf

from tensorflow.python.client import device_lib
from tensorflow_on_slurm import tf_config_from_slurm


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def get_available_devices():
    
    local_device_protos = device_lib.list_local_devices()
    
    return [x.name for x in local_device_protos]


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def main(_):
    
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    print()
    cluster, my_job_name, my_task_index = tf_config_from_slurm(ps_number=1)
    print()
    
    cluster_spec = tf.train.ClusterSpec(cluster)
    
    server = tf.train.Server(
        server_or_cluster_def=cluster_spec,
        job_name=my_job_name,
        task_index=my_task_index
    )
    
    if (my_job_name == 'ps'):

        print('Joining server...')
              
        server.join()
        sys.exit(0)

    
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    loss = tf.reduce_mean(cross_entropy)
    opt = tf.train.AdamOptimizer(1e-3)
    
    opt = tf.train.SyncReplicasOptimizer(
        opt,
        replicas_to_aggregate=len(cluster['worker']),
        total_num_replicas=len(cluster['worker'])
    )
    
    is_chief           = (my_task_index == 0)
    sync_replicas_hook = opt.make_session_run_hook(is_chief)
    train_step         = opt.minimize(loss, global_step)

    
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    sync_replicas_hook = opt.make_session_run_hook(is_chief)
    
    sess = tf.train.MonitoredTrainingSession(
        master=server.target,
        is_chief=is_chief,
        hooks=[sync_replicas_hook]
    )

    batch_size = 64
    max_epoch = 10000

    print("--- Training ---")
    
    for i in range(max_epoch):
        
        batch = mnist.train.next_batch(batch_size)
        
        if i % len(cluster['worker']) != my_task_index:
            continue
            
        _, train_accuracy, xentropy = sess.run(
            [train_step, accuracy, cross_entropy],
            feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5}
        )
        
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    return



#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

if (__name__ == "__main__"):

    main()
    
    pass

