
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

import json

import python_utils as pyutils
import keras_utils as  kutils
from python_utils.python_imports import *
from keras_utils.keras_imports   import *

print()
pyutils.printVersions()
print()
kutils.printVersions()
print()


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

pp = pprint.PrettyPrinter(depth=4)

def main():

    args = parse_arguments()
    
    file_input     = args.file_input
    file_truth     = args.file_truth
    n_timesteps    = args.n_timesteps
    n_outputs      = args.n_outputs
    n_events_train = args.n_events_train
    n_epochs       = args.n_epochs
    layers_hidden  = args.layers_hidden
    loss           = args.loss
    optimizer      = args.optimizer
    useGPU         = args.useGPU
    
    print()
    print("file_input:    " + str(file_input) )
    print("file_truth:    " + str(file_truth) )
    print()

    assert(os.path.exists(file_input))
    assert(os.path.exists(file_truth))

    useGPU = False
    
    #if (arg_useGPU != 0):
        
    #    useGPU = True
        

        
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    #printVersions()

    
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    if (useGPU is True):
        
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        code   = result.returncode
    
        if (code != 0):
            
            print("Error! CUDA not available")
            print("'nvidia-smi' gave nonzero return code " + str(code) + "\n") 
                  
            return
    

    
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    model_name  = 's2waveforms-xy'

    n_channels  = 127
    n_inputs    = n_timesteps * n_channels
    layers_desc = 'layers' + str(n_inputs) + '-' + str('-').join(str(x) for x in layers_hidden) + '-' + str(n_outputs)


    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------

    print("Loading training data...")
    
    train_data  = np.load(file_input)
    train_truth = np.load(file_truth)
    
    train_data  = train_data [0:n_events_train, :]
    train_truth = train_truth[0:n_events_train, :]
    
    
    #--------------------------------------------------------------------------
    # Training Data
    #--------------------------------------------------------------------------
    
    print()
    print("Channels:             " + str(n_channels))
    print("Timesteps:            " + str(n_timesteps))
    print("Outputs:              " + str(n_outputs) )
    print("Training Input shape: " + str(train_data.shape ))
    print("Training Truth shape: " + str(train_truth.shape))
    print()
    
    
    #--------------------------------------------------------------------------
    # Initialize Model
    #--------------------------------------------------------------------------
    
    print("Loading model...")
    
    model = kutils.dnnModel(
        n_inputs,
        n_outputs,
        layers_hidden,
        'elu',
        loss,
        optimizer,
        keep_rate=0.00005)
    
    print()
    print("Model Summary:")
    model.summary()
    print()
    

    #--------------------------------------------------------------------------
    # Fit Model
    #  to do: reset model 
    #--------------------------------------------------------------------------
    
    print("Training model...")
    
    history = model.fit(
        train_data,
        train_truth,
        batch_size=64,
        #batch_size=32,
        epochs=n_epochs,
        verbose=2
    )
    
    print("Training complete.")
    
    
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------

    dct_config  = model.get_config()
    dct_history = history.history
    
    last_loss = dct_history['loss'][n_epochs-1]
    last_loss = int(round(last_loss*100, 0))
    last_loss = 'loss' + str(last_loss)
    
    last_acc  = dct_history['acc'][n_epochs-1]
    last_acc  = int(round(last_acc*1e4, 0))
    last_acc  = 'ac%04d' % last_acc
    
    
    #--------------------------------------------------------------------------
    # Save Model
    #--------------------------------------------------------------------------
    
    loss_desc = loss
    
    if ('_' in loss_desc):
        
        loss_desc = ''.join( [x[:1] for x in loss.split('_') ] )
    
    desc     = 'dnn_' + model_name + '_'
    desc     += ("ts%04d" % n_timesteps) + '_' 
    desc     += ('e%02d' % n_epochs) + '_' 
    desc     += loss_desc + '_' 
    desc     += optimizer + '_'
    desc     += last_acc + '_'
    desc     += layers_desc

    folder   = "models/cpu/"
    
    if (useGPU is True):
        
        folder   = "models/gpu/"

    name_h5  = folder + desc + '.h5'
    name_cfg = folder + desc + '_cfg.json'
    name_hst = folder + desc + '_hist.json'
    name_png = folder + desc + '.png'
    
    
    #--------------------------------------------------------------------------
    # Save
    #--------------------------------------------------------------------------
    
    model.save(name_h5, overwrite=True)
    
    with open(name_cfg, 'w') as fp: json.dump(dct_config , fp)
    with open(name_hst, 'w') as fp: json.dump(dct_history, fp)

    print("\nSaved model: '" + name_h5 + "\n")
    

    #--------------------------------------------------------------------------
    # Predict
    #--------------------------------------------------------------------------
    
    test_data  = np.load(file_input)
    test_truth = np.load(file_truth)
    
    test_data  = test_data [n_events_train:, :]
    test_truth = test_truth[n_events_train:, :]
    
    print("Test Input shape: " + str(test_data.shape ))
    print("Test Truth shape: " + str(test_truth.shape))
    
    arr_pred = model.predict(test_data)

    
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    df_out           = pd.DataFrame()
    df_out['x_pred'] = arr_pred[:, 0]
    df_out['y_pred'] = arr_pred[:, 1]
    df_out['x_true'] = test_truth[:, 0]
    df_out['y_true'] = test_truth[:, 1]

    dir_pred = "predictions/cpu/"
    
    if (useGPU is True): 
        
        print("HERE")
        dir_pred = "predictions/gpu/"

    file_hdf = dir_pred + os.path.basename(name_h5).replace('.h5', '.hdf5')


    try:
    
        df_out.to_hdf(file_hdf, 'df')
        print("\nSaved predictions: '" + file_hdf + "'\n")

    except Exception as e:
        
        print("\nException saving prediction to: '" + file_hdf + "'")
        print(e)

    
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    #try:
    #    
    #    plot_model(model, to_file=name_png, show_layer_names=True, show_shapes=True)
    #    
    #except Exception as ex:
    #    
    #    print("\nException saving '" + name_png + str(ex) + "\n")

        
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------

    return


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument('-file_input'    , required=True)
    parser.add_argument('-file_truth'    , required=True)
    parser.add_argument('-n_timesteps'   , required=True, type=int)
    parser.add_argument('-n_outputs'     , required=True, type=int)
    parser.add_argument('-n_events_train', required=True, type=int)
    parser.add_argument('-n_epochs'      , required=True, type=int)
    parser.add_argument('-layers_hidden' , required=True, type=int, nargs="+")
    parser.add_argument('-loss'          , required=True)
    parser.add_argument('-optimizer'     , required=True)
    parser.add_argument('-useGPU'        , required=True, type=int)

    arguments = parser.parse_args()
    
    return arguments


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

if __name__ == "__main__":

    t0 = time.time()
    
    main()
    
    t1 = time.time()
    dt = round(t1 - t0, 0)
    dt = datetime.timedelta(seconds=dt)
    
    print("Total Time: " + str(dt) + "\n")


