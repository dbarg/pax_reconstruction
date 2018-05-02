####################################################################################################
####################################################################################################

import sys
import glob
import os.path
import time

import numpy as np
import pandas as pd
import skimage.measure as skimgmeas

from IPython.display import clear_output
from IPython.display import display

sys.path.append(os.path.abspath("../../"))
from pax_utils import waveform_utils
from pax_utils import performance_utils


####################################################################################################
####################################################################################################

def get_data(dir_input_s2, df_events, s2_window_max, resample_factor=1):
 
    ####################################################################################################
    ####################################################################################################

    if (s2_window_max % resample_factor != 0):

        print(s2_window_max)
        print(resample_factor)
    
    assert(s2_window_max % resample_factor == 0)

    resample                = False
    s2_window_max_resampled = s2_window_max
    
    if (resample_factor > 1):

        resample                = True
        s2_window_max_resampled = s2_window_max / resample_factor
        
    
    ####################################################################################################
    # Input directory - containing S2 waveforms for all top channels
    ####################################################################################################
    
    dir_format_s2   = dir_input_s2 + '/' + 'event' + ('[0-9]' * 6) + '_S2waveforms.pkl'
    lst_contents_s2 = glob.glob(dir_format_s2)
    lst_events      = df_events['event_number'].as_matrix().tolist()
    
    
    ####################################################################################################
    # Input data shape: (1, 10, 127)
    # Truth data shape: (1, 2)
    ####################################################################################################
    
    nEvents    = len(df_events.index)
    n_channels = 127
    
    
    ####################################################################################################
    ####################################################################################################
        
    train_data           = np.zeros((nEvents, s2_window_max*n_channels))
    train_data_resampled = np.zeros((nEvents, s2_window_max_resampled*n_channels))
    train_truth          = np.zeros((nEvents, 2))
    
    event_train_truth = np.zeros(2)

                   
     
    #return train_data, train_truth
    
    
    ####################################################################################################
    # Loop over events
    ####################################################################################################
    
    nEmpty = 0
    
    iIndex = 0

    for iEvent, event_num in enumerate(lst_events):
        
        t0 = time.time()
        
        ################################################################################################
        # Get Event Information
        ################################################################################################
        
        infile    = dir_input_s2 + '/event' + format(event_num, '06d') + '_S2waveforms.pkl'

        
        ################################################################################################
        ################################################################################################
        
        df_event        = df_events.iloc[iEvent]
        event_s2_count  = df_event.loc['event_s2_count' ]
        event_s2_length = df_event.loc['event_s2_length']
        event_s2_left   = df_event.loc['event_s2_left'  ]
        event_s2_right  = df_event.loc['event_s2_right' ]
        intr_count      = df_event.loc['intr_count'     ]
        x_true          = df_event.loc['x'              ]
        y_true          = df_event.loc['y'              ]
     
        
        ################################################################################################
        ################################################################################################
        
        if (intr_count < 1):
            
            nEmpty += 1
        
            continue
            
        
        ################################################################################################
        # Get S2 waveforms for top channels
        ################################################################################################
        
        df_s2waveforms = pd.read_pickle(infile)
        df_s2waveforms = waveform_utils.addEmptyChannelsToDataFrame(df_s2waveforms)
    
        
        ################################################################################################
        ################################################################################################
        
        for iChannel, row in df_s2waveforms.iterrows():
            
            channel          = row['channel']
            channel_left     = row['left']
            channel_right    = row['right']
            channel_length   = row['length']
            channel_integral = row['sum']
            channel_raw_data = row['raw_data']
        
            if (row['raw_data'] is not None):
                
                assert(channel_length == row['raw_data'].size)
            
                 
            ############################################################################################
            # Pad the S2 array for all channels in the event
            ############################################################################################
            
            arr_channel = np.zeros(event_s2_length)
            
            if (channel_length > 0):
                
                channel_left_offset  = channel_left   - event_s2_left
                channel_right_offset = event_s2_right - channel_right
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
            
            
                assert(channel_left    >= event_s2_left )
                assert(channel_right   <= event_s2_right)
                assert(event_s2_length == channel_left_offset + channel_length + channel_right_offset  )

                arr_channel[channel_left_offset : channel_left_offset + channel_length] = channel_raw_data  
                
                assert( abs(channel_integral - np.sum(arr_channel)) < 1e-4 )
    
            
            ############################################################################################
            # Pad to the widest S2 over all events
            ############################################################################################
                
            arr_channel_padded                       = np.zeros(s2_window_max)
            arr_channel_padded[0 : arr_channel.size] = arr_channel
            
             
            ############################################################################################
            ############################################################################################
            
            i0 = iChannel*s2_window_max
            i1 = i0 + s2_window_max
            
            train_data[iEvent, i0:i1] = arr_channel_padded
        

            ############################################################################################
            ############################################################################################
            
            if (resample):
                
                arr_resampled = skimgmeas.block_reduce(arr_channel_padded, block_size=(resample_factor,), func=np.sum)
                
                i0_r = iChannel*s2_window_max_resampled
                i1_r = i0_r + s2_window_max_resampled
               
                #print("Resampled by factor: " + str(resample_factor))
                #print(arr_resampled.shape)
                #print(train_data_resampled.shape)
                
                train_data_resampled[iEvent, i0_r:i1_r] = arr_resampled
            
            
                ########################################################################################
                ########################################################################################
            
                if (resample_factor == s2_window_max):
                
                    str_chan   = 's2_area_%03d' % iChannel
                    
                    s2integral_df  = df_event[:][str_chan].astype('float32')
                    s2integral_arr = arr_resampled[0].astype('float32')
                        
                    assert(arr_resampled.size == 1)
                    
                    relerr = abs(s2integral_df - s2integral_arr)
                    
                    if (relerr > 0):
                        
                        relerr /= s2integral_df
                        
                    eq = relerr < 0.1
                    
                    if (not eq):
                     
                        print()
                        print(event_num)
                        print(s2integral_df)
                        print(s2integral_arr)
                        print(relerr)

                    #assert(eq)
                
                
            ############################################################################################
            # End loop over channels
            ############################################################################################
            
            continue
    
        
        ################################################################################################
        ################################################################################################
        
        event_train_truth[0]      = x_true
        event_train_truth[1]      = y_true
        train_truth[iEvent, :]    = event_train_truth
        
        #print("x, y: " + str(x_true) + ", " + str(y_true))
        
        
        ################################################################################################
        # Sanity
        ################################################################################################
        
        df_event_s2s  = df_s2waveforms[df_s2waveforms['left'] != 0].copy(deep=True)
        df_event_s2s.reset_index(inplace=True, drop=True)
        
        min_left      = np.amin( df_event_s2s.left.as_matrix()  )
        max_right     = np.amax( df_event_s2s.right.as_matrix() )
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
        
        
        ################################################################################################
        # End loop over events
        ################################################################################################
    
        t1 = time.time()
    
        performance_utils.time_event(iEvent, event_num, t1 - t0)
        
        continue

        
    ####################################################################################################
    ####################################################################################################

    print()
    print(str(nEmpty) + " empty events")
    print()
    print("Input shape:     " + str(train_data.shape))
    print("Resampled shape: " + str(train_data_resampled.shape))
    
          
    ####################################################################################################
    ####################################################################################################
    
    if (resample):
        
        return train_data_resampled, train_truth
        
    return train_data, train_truth


####################################################################################################
####################################################################################################

def getEventsDataFrame(input_path):

    ################################################################################################
    ################################################################################################

    events_df = pd.read_pickle(input_path)
    #events_df = events_df[events_df['intr_count'] == 1] 

    
    ################################################################################################
    ################################################################################################
    
    arr_s2_right_minus_left = (events_df['event_s2_right'] - events_df['event_s2_left']).as_matrix()
    arr_s2_length           =  events_df['event_s2_length'].as_matrix()  
    
    s2_window_max = np.amax(arr_s2_right_minus_left) + 1
    s2_length_max = np.amax(arr_s2_length)
    
    test = s2_window_max == s2_length_max
    
    assert(test)
    

    
    ####################################################################################################
    ####################################################################################################
    
    return events_df, s2_window_max
    