
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

import argparse
import sys
import gc
import glob
import math
import numpy as np
import pandas as pd
import os.path
import pickle
import skimage.measure as skimgmeas
import time

from IPython.display import clear_output
from IPython.display import display

from pax_utils import waveform_utils
from pax_utils import performance_utils
from pax_utils import s1s2_utils
from pax_utils import numeric_utils


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

cols = [
    'event_number',
    'event_s2_count',
    'event_s2_length',
    'event_s2_left',
    'event_s2_right',
    'intr_count',
    'x',
    'y',
    'intr_x',
    'intr_y',
    'intr_z',
    'intr_drift_time',
    'intr_x_nn',
    'intr_y_nn'
]


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

iEventStart  = 0
nEventsTrain = 200000
iEventEnd    = iEventStart + nEventsTrain


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def get_data(dir_input_s2, input_path, dir_output, resample_factor=1):
 
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    print("Input directory: " + dir_input_s2)
    print("Resample factor: " + str(resample_factor))


    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    df_events     = pd.read_pickle(input_path)
    df_events     = df_events[df_events['intr_count'] == 1].reset_index(drop=True)
    s2_window_max = np.amax(df_events['event_s2_length'].values )
    s2_window_max = int(math.ceil(s2_window_max / 100.0)) * 100
    TotalEvents   = len(df_events.index)
    df_events     = df_events[iEventStart:iEventEnd][:]

    
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    assert (s2_window_max < 2000)
    s2_window_max = 2000
    
    nTimesteps   = s2_window_max / resample_factor
    df_events    = df_events[:][cols]
        
    print("S2 Window Max: " + str(s2_window_max))
    print("Timesteps:     " + str(nTimesteps))
    print(TotalEvents)

    file_out_input = 'array_train_input_events%06d-%06d_timesteps%04d' % (iEventStart, iEventEnd - 1, nTimesteps)
    file_out_truth = 'array_train_truth_events%06d-%06d_timesteps%04d' % (iEventStart, iEventEnd - 1, nTimesteps)

    file_out_input = dir_output + '/' + file_out_input
    file_out_truth = dir_output + '/' + file_out_truth

    print()
    print("Total Events:      " + str(TotalEvents))
    print("Processing Events: " + str(nEventsTrain))
    print("Start Event:       " + str(iEventStart))
    print("End Event:         " + str(iEventEnd))
    print("Shape:             " + str(df_events.shape))
    print("Max S2 Window:     " + str(s2_window_max))
    print("file_out_input:    " + file_out_input)
    print("file_out_truth:    " + file_out_truth)
    print()
    
    assert(s2_window_max % resample_factor == 0)


    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------

    if (s2_window_max % resample_factor != 0):

        print(s2_window_max)
        print(resample_factor)
    
    assert(s2_window_max % resample_factor == 0)

    resample                = False
    s2_window_max_resampled = s2_window_max
    
    if (resample_factor > 1):

        resample                = True
        s2_window_max_resampled = s2_window_max / resample_factor

        s2_window_max_resampled = int(s2_window_max_resampled)
        #print(s2_window_max_resampled)    

    
    #--------------------------------------------------------------------------
    # Input directory - containing S2 waveforms for all top channels
    #--------------------------------------------------------------------------
    
    dir_format_s2   = dir_input_s2 + '/' + 'event' + ('[0-9]' * 7) + '_S2waveforms.pkl'
    lst_contents_s2 = glob.glob(dir_format_s2)
    #lst_events      = df_events['event_number'].as_matrix().tolist()
    lst_events      = df_events['event_number'].values.tolist()
    
    
    #--------------------------------------------------------------------------
    # Input data shape: (1, 10, 127)
    # Truth data shape: (1, 2)
    #--------------------------------------------------------------------------
    
    nEvents    = len(df_events.index)
    n_channels = 127
    
    
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
        
    #train_data           = np.zeros((nEvents, s2_window_max*n_channels)          , dtype ='float32')
    train_data_resampled = np.zeros((nEvents, s2_window_max_resampled*n_channels), dtype ='float32')
    train_truth          = np.zeros((nEvents, 2)                                 , dtype ='float32')
    event_train_truth    = np.zeros(2                                            , dtype ='float32')
    arr_temp             = np.zeros(s2_window_max*n_channels                     , dtype ='float32')
    arr_temp_resampled   = np.zeros(s2_window_max_resampled*n_channels           , dtype ='float32')
    
    
    #--------------------------------------------------------------------------
    # Loop over events
    #--------------------------------------------------------------------------
    
    nEmpty = 0
    iIndex = 0

    for iEvent, event_num in enumerate(lst_events):
        
        if (iEvent % 100 == 0 or iEvent == 0):         
            print("Event: " + str(iEvent))

        t0 = time.time()
        
        #gc.collect() # time consuming
        
        
        #--------------------------------------------------------------------------
        # Get Event Information
        #--------------------------------------------------------------------------
        
        infile = dir_input_s2 + '/event' + format(event_num, '07d') + '_S2waveforms.pkl'

        
        #--------------------------------------------------------------------------
        #--------------------------------------------------------------------------
        
        df_event        = df_events.iloc[iEvent]
        event_s2_count  = df_event.loc['event_s2_count' ]
        event_s2_length = df_event.loc['event_s2_length']
        event_s2_left   = df_event.loc['event_s2_left'  ]
        event_s2_right  = df_event.loc['event_s2_right' ]
        intr_count      = df_event.loc['intr_count'     ]
        x_true          = df_event.loc['x'              ]
        y_true          = df_event.loc['y'              ]
    
        event_train_truth[0]   = x_true
        event_train_truth[1]   = y_true
        train_truth[iEvent, :] = event_train_truth

        
        #--------------------------------------------------------------------------
        #--------------------------------------------------------------------------
        
        if (intr_count < 1):
            
            nEmpty += 1
        
            continue
            
        
        #--------------------------------------------------------------------------
        # Get S2 waveforms for top channels
        #--------------------------------------------------------------------------

        df_s2waveforms = pd.DataFrame()
        df_s2waveforms = pd.read_pickle(infile)
        df_s2waveforms = waveform_utils.addEmptyChannelsToDataFrame(df_s2waveforms)
        
        t1 = time.time()
        
        print("HERE")

        #--------------------------------------------------------------------------
        #--------------------------------------------------------------------------
        
        for iChannel, ser_channel in df_s2waveforms.iterrows():
            
            #----------------------------------------------------------------------
            # This piece takes ~ 0.7 s
            #----------------------------------------------------------------------
    
            arr_channel_padded = get_padded_array(
                ser_channel,
                s2_window_max,
                event_s2_left,
                event_s2_right,
                event_s2_length)
          
            
            #----------------------------------------------------------------------
            # This piece takes ~ 0.3 s
            #----------------------------------------------------------------------
            
            if (not resample):
            
                #------------------------------------------------------------------
                #------------------------------------------------------------------
            
                i0 = iChannel * s2_window_max
                i1 = i0       + s2_window_max
    
                #arr_temp[i0:i1] = np.array(arr_channel_padded, copy=True)
                arr_temp[i0:i1] = arr_channel_padded

            else:
            
                #------------------------------------------------------------------
                #------------------------------------------------------------------
                
                arr_resampled = skimgmeas.block_reduce(arr_channel_padded, block_size=(resample_factor,), func=np.sum)
                i0_r          = iChannel * s2_window_max_resampled
                i1_r          = i0_r     + s2_window_max_resampled
               
                arr_temp_resampled[i0_r : i1_r] = arr_resampled

                
                #------------------------------------------------------------------
                #------------------------------------------------------------------
                
                #print("Resampled by factor: " + str(resample_factor))
                #print(arr_resampled.shape)
                #print(arr_temp_resampled.shape)
                #print(train_data_resampled.shape)            
                #print()
                
                
            #----------------------------------------------------------------------
            # End loop over channels
            #----------------------------------------------------------------------

            continue
    
    
        #--------------------------------------------------------------------------
        #--------------------------------------------------------------------------
        
        t2 = time.time()
        
        #t1 = time.time()
        #performance_utils.time_event(iEvent, event_num, t1 - t0)

       
        #--------------------------------------------------------------------------
        # *** Memory grows here ***
        #--------------------------------------------------------------------------
        
        #continue
        
        if (not resample):
            
            #train_data[iEvent] = np.array(arr_temp, copy=True)
            train_data[iEvent] = arr_temp
            
        else:
            
            #train_data_resampled[iEvent] = np.array(arr_temp_resampled, copy=True)
            train_data_resampled[iEvent] = arr_temp_resampled
            
        #continue
        
        t3 = time.time()
        
        
        #--------------------------------------------------------------------------
        #--------------------------------------------------------------------------
        
        if (resample_factor == s2_window_max):
        
            #arr_s2integrals_evt   = df_event[s1s2_utils.getS2integralsDataFrameColumns()].as_matrix()
            arr_s2integrals_evt   = df_event[s1s2_utils.getS2integralsDataFrameColumns()].values
            arr_s2integrals_wfs   = train_data_resampled[iEvent, :]
            arr_s2integrals_equal = numeric_utils.compareArrays(arr_s2integrals_evt, arr_s2integrals_wfs, 0.2)
                
            if (not arr_s2integrals_equal):
                
                print("event: " + str(event_num))
                print(arr_s2integrals_evt.size)
                print(arr_s2integrals_evt)
                print()
                print(arr_s2integrals_wfs.size)
                print(arr_s2integrals_wfs)
            
            assert(arr_s2integrals_equal)
        
        t4 = time.time()
        
        
        #--------------------------------------------------------------------------
        # Sanity
        #--------------------------------------------------------------------------
        
        #assert(train_data.shape[1] == arr_temp.size)
        #assert(train_data.shape[1] == s2_window_max*n_channels)
        
        df_event_s2s  = df_s2waveforms[df_s2waveforms['left'] != 0].copy(deep=True)
        df_event_s2s.reset_index(inplace=True, drop=True)
        
        #min_left      = np.amin( df_event_s2s.left.as_matrix()  )
        #max_right     = np.amax( df_event_s2s.right.as_matrix() )
        min_left      = np.amin( df_event_s2s.left.values )
        max_right     = np.amax( df_event_s2s.right.values)
        max_length    = np.amax( df_event_s2s.length            )
      
        test1 = min_left   >= event_s2_left  
        test2 = max_right  <= event_s2_right 
        test3 = max_length <= event_s2_length
        
        if (not test1 or not test2 or not test3):
            
            print("event:      " + str(event_num)      )
            print("left df:    " + str(min_left)       )
            print("left evt:   " + str(event_s2_left)  )
            print("right df:   " + str(max_right)      )
            print("right evt:  " + str(event_s2_right) )
            print("length df:  " + str(max_length)     )
            print("length evt: " + str(event_s2_length))
            
        assert (test1)
        assert (test2)
        assert (test3)
        
        
        #--------------------------------------------------------------------------
        # End loop over events
        #--------------------------------------------------------------------------

        t5   = time.time()
        dt54 = round(t5 - t4, 2)
        dt43 = round(t4 - t3, 2)
        dt32 = round(t3 - t2, 2)
        dt21 = round(t2 - t1, 2)
        dt10 = round(t1 - t0, 2)
        dt50 = round(t5 - t0, 2)
        
        performance_utils.time_event(iEvent, event_num, t5 - t0)

        if (False):
            
            print(" -> IO:           " + str(dt10))
            print(" -> channel loop: " + str(dt21))
            print(" -> array assign: " + str(dt32))
            print(" -> resampling:   " + str(dt43))
            print(" -> Sanity:       " + str(dt54))
            print(" -> Total:        " + str(dt50))
            print()
        
        
        #--------------------------------------------------------------------------
        #--------------------------------------------------------------------------
        
        #break
        continue

        
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------

    print()
    print(str(nEmpty) + " empty events")
    print()
    #print("Input shape:     " + str(train_data.shape))
    print("Train: " + str(train_data_resampled.shape))
    print("Truth: " + str(train_truth.shape))
    
    #if (resample):
    #    train_data = train_data_resampled
    
    np.save(file_out_input, train_data_resampled)
    np.save(file_out_truth, train_truth)
    
    return

    #test_train_data  = np.load(file_out_input + '.npy')
    #test_train_truth = np.load(file_out_truth + '.npy')
    
         
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
    
    #if (resample):
    #    
    #    return train_data_resampled, train_truth
    #
    #return train_data, train_truth
    
    
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def get_padded_array(
    ser_channel,
    s2_window_max,
    event_s2_left,
    event_s2_right,
    event_s2_length):
 
    channel          = ser_channel['channel']
    channel_left     = ser_channel['left']
    channel_right    = ser_channel['right']
    channel_length   = ser_channel['length']
    channel_integral = ser_channel['sum']
    channel_raw_data = ser_channel['raw_data']
    
    if (ser_channel['raw_data'] is not None):
        
        assert(channel_length == ser_channel['raw_data'].size)

        
    #--------------------------------------------------------------------------
    # Pad the S2 array for all channels in the event
    #--------------------------------------------------------------------------
    
    event_s2_left   = int(event_s2_left)
    event_s2_right  = int(event_s2_right)
    event_s2_length = int(event_s2_length)

    arr_channel        = np.zeros(event_s2_length, dtype ='float32')
    arr_channel_padded = np.zeros(s2_window_max  , dtype ='float32')
    
    if (channel_length > 0):
        
        #--------------------------------------------------------------------------
        #--------------------------------------------------------------------------
        
        channel_left_offset  = channel_left   - event_s2_left
        channel_right_offset = event_s2_right - channel_right
        
        assert(channel_left    >= event_s2_left )
        assert(channel_right   <= event_s2_right)
        assert(event_s2_length == channel_left_offset + channel_length + channel_right_offset  )
    
        arr_channel[channel_left_offset : channel_left_offset + channel_length] = channel_raw_data

        channels_sum   = np.sum(arr_channel)
        integrals_diff = abs(channel_integral - channels_sum)
        integrals_eq   = integrals_diff < 1e-2

        if (not integrals_eq):
            
            print()
            print(" -> channel:     " + channel         ) 
            print(" -> event:       " + channel_integral)
            print(" -> channls sum: " + channels_sum    )
            print(" -> difference:  " + integrals_diff  )
            print()

        #assert( abs(channel_integral - np.sum(arr_channel)) < 1e-4 )

        
        #--------------------------------------------------------------------------
        #--------------------------------------------------------------------------
        
        
        #channel_length       = channel_left_offset + channel_length + channel_right_offset
        #print()
        #print("s2 window all chan:   " + str(event_s2_length))
        #print("channel length:       " + str(channel_length))
        #print("channel left offset:  " + str(channel_left_offset))
        #print("channel right offset: " + str(channel_right_offset))
        #print("event:       " + str(event_num)      )
        #print("left chan:   " + str(channel_left)       )
        #print("left evt:    " + str(event_s2_left)  )
        #print("right chan:  " + str(channel_right)      )
        #print("right evt:   " + str(event_s2_right) )
        #print("length chan: " + str(channel_length)     )
        #print("length evt:  " + str(event_s2_length))
    
    
    #--------------------------------------------------------------------------
    # Pad to the widest S2 over all events
    #--------------------------------------------------------------------------
        
    arr_channel_padded[0 : arr_channel.size] = arr_channel
    
  

    return arr_channel_padded


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def getEventsDataFrame(input_path):

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------

    events_df = pd.read_pickle(input_path)
    #events_df = events_df[events_df['intr_count'] == 1] 

    
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    #arr_s2_right_minus_left = (events_df['event_s2_right'] - events_df['event_s2_left']).as_matrix()
    #arr_s2_length           =  events_df['event_s2_length'].as_matrix()  
    arr_s2_right_minus_left = (events_df['event_s2_right'] - events_df['event_s2_left']).values
    arr_s2_length           =  events_df['event_s2_length'].values  
    
    s2_window_max = np.amax(arr_s2_right_minus_left) + 1
    s2_length_max = np.amax(arr_s2_length)
    
    test = s2_window_max == s2_length_max
    
    assert(test)
    

    
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    return events_df, s2_window_max
    

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument('-dir_input'      , required=True)
    parser.add_argument('-dir_output'     , required=True)
    parser.add_argument('-input_path'     , required=True)
    parser.add_argument('-resample_factor', required=True, type=int)

    arguments = parser.parse_args()
    
    return arguments


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def main():
    
    print("main")

    args = parse_arguments()

    dir_input       = args.dir_input
    dir_output      = args.dir_output
    input_path      = args.input_path
    resample_factor = args.resample_factor

    #print(dir_input)
    #print(resample_factor)
    
    get_data(dir_input, input_path, dir_output, resample_factor)
    
    return


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

if __name__== "__main__":
    
    main()
