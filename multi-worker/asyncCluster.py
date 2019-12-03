#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

import argparse
import keras
import sys
import tensorflow as tf

from tensorflow.python.client import device_lib

FLAGS = None


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def main(_):

    print(type(FLAGS.worker_hosts))
    print(FLAGS.worker_hosts)
    
    FLAGS.worker_hosts = FLAGS.worker_hosts.replace('[', '')
    FLAGS.worker_hosts = FLAGS.worker_hosts.replace(']', '')

    ps_hosts     = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    print("->{0}<-".format(FLAGS.worker_hosts))
    print("PS hosts:     {0}".format(ps_hosts))
    print("Worker hosts: {0}".format(worker_hosts))
    
    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
  
    # Create and start a server for the local task.
    server = tf.train.Server(cluster, job_name=FLAGS.job_name,task_index=FLAGS.task_index)
  
    if FLAGS.job_name == "ps":
        
        if (False):
            
            # create a shared queue on the parameter server which is visible on /job:ps/task:%d
            with tf.device('/job:ps/task:%d' % FLAGS.task_index):
                queue = tf.FIFOQueue(cluster.num_tasks('worker'), tf.int32, shared_name='done_queue%d' % FLAGS.task_index)
    
            # wait for the queue to be filled
            with tf.Session(server.target) as sess:
                for i in range(cluster.num_tasks('worker')):
                    sess.run(queue.dequeue())
                    print('ps:%d received "done" from worker:%d' % (FLAGS.task_index, i))
                print('ps:%d quitting' % FLAGS.task_index)
                
        else:
            server.join()
        
        
    elif FLAGS.job_name == "worker":
  
      # Assigns ops to the local worker by default.
      with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index,cluster=cluster)):
  
        print("worker: {0}, task={1}".format(worker_device, task_index))
        # Build model...
        loss = 13
        global_step = tf.contrib.framework.get_or_create_global_step()
  
        train_op = tf.train.AdagradOptimizer(0.01).minimize(loss, global_step=global_step)
  
      # The StopAtStepHook handles stopping after running given steps.
      hooks=[tf.train.StopAtStepHook(last_step=1)]
  
      # The MonitoredTrainingSession takes care of session initialization,
      # restoring from a checkpoint, saving to a checkpoint, and closing when done
      # or an error occurs.
      with tf.train.MonitoredTrainingSession(master=server.target,
                                             is_chief=(FLAGS.task_index == 0),
                                             checkpoint_dir="/tmp/train_logs",
                                             hooks=hooks) as mon_sess:
        while not mon_sess.should_stop():
          # Run a training step asynchronously.
          # See `tf.train.SyncReplicasOptimizer` for additional details on how to
          # perform *synchronous* training.
          # mon_sess.run handles AbortedError in case of preempted PS.
          mon_sess.run(train_op)
        
       
    
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

if   __name__ == "__main__":
    
    print("Async MAIN")
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    # Flags for defining the tf.train.ClusterSpec
    
    parser.add_argument(
        "--ps_hosts",
        type=str,
        default="",
        help="Comma-separated list of hostname:port pairs"
    )
    
    parser.add_argument(
        "--worker_hosts",
        type=str,
        default="",
        help="Comma-separated list of hostname:port pairs"
    )
    
    parser.add_argument(
        "--job_name",
        type=str,
        default="",
        help="One of 'ps', 'worker'"
    )
    
    # Flags for defining the tf.train.Server
    parser.add_argument(
        "--task_index",
        type=int,
        default=0,
        help="Index of task within the job"
    )
    
    FLAGS, unparsed = parser.parse_known_args()
    
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)