#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

import glob
import os.path
import numpy as np
import pandas as pd
import sys

from IPython.display import clear_output
from IPython.display import display

from pax_utils import waveform_utils


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def get_data(df_events, s2_window_max):
 
    #--------------------------------------------------------------------------
    # Input directory - containing S2 waveforms for all top channels
    #--------------------------------------------------------------------------
    
    dir_input_s2    = "../../pax_merge/merged/waveforms_test/s2"
    dir_format_s2   = dir_input_s2 + '/' + 'event' + ('[0-9]' * 4) + '_S2waveforms.pkl'
    lst_contents_s2 = glob.glob(dir_format_s2)
    lst_events      = df_events['event_number'].as_matrix().tolist()
    
    
    #--------------------------------------------------------------------------
    # Input data shape: (1, 10, 127)
    # Truth data shape: (1, 2)
    #--------------------------------------------------------------------------
    
    nEvents    = len(df_events.index)
    n_channels = 127
    
    
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
        
    train_data        = np.zeros((nEvents, s2_window_max, n_channels))
    train_truth       = np.zeros((nEvents, 2))
    event_train_data  = np.zeros((s2_window_max, n_channels))
    event_train_truth = np.zeros(2)
                     
                             
    #--------------------------------------------------------------------------
    # Loop over events
    #--------------------------------------------------------------------------
    
    nEmpty = 0
    
    for iEvent, event_num in enumerate(lst_events):
        
        
        #----------------------------------------------------------------------
        # Get Event Information
        #----------------------------------------------------------------------
        
        infile = dir_input_s2 + '/event' + format(event_num, '04d') + '_S2waveforms.pkl'
        
        df_event        = df_events.iloc[iEvent]
        event_s2_count  = df_event.loc['event_s2_count' ]
        event_s2_length = df_event.loc['event_s2_length']
        event_s2_left   = df_event.loc['event_s2_left'  ]
        event_s2_right  = df_event.loc['event_s2_right' ]
        intr_count      = df_event.loc['intr_count'     ]
        x_true          = df_event.loc['x'              ]
        y_true          = df_event.loc['y'              ]
       
        print(" -> Event Index, Event Number: " + str(iEvent) + ", " + str(event_num))
        clear_output(wait=True)
        
        
        #----------------------------------------------------------------------
        #----------------------------------------------------------------------
        
        if (intr_count < 1):
            
            nEmpty += 1
        
            continue
            
        
        #----------------------------------------------------------------------
        # Get S2 waveforms for top channels
        #----------------------------------------------------------------------
        
        df_s2waveforms = pd.read_pickle(infile)
        df_s2waveforms = waveform_utils.addEmptyChannelsToDataFrame(df_s2waveforms)
    
        for iChannel, row in df_s2waveforms.iterrows():
            
            channel          = row['channel']
            channel_left     = row['left']
            channel_right    = row['right']
            channel_length   = row['length']
            channel_integral = row['sum']
            channel_raw_data = row['raw_data']
        
            assert(channel_length == channel_raw_data.size)
            
                 
            #------------------------------------------------------------------
            # Pad the S2 array for all channels in the event
            #------------------------------------------------------------------
            
            arr_channel = np.zeros(event_s2_length)
            
            if (channel_length > 0):
                
                channel_left_offset  = channel_left   - event_s2_left
                channel_right_offset = event_s2_right - channel_right
                #channel_length       = channel_left_offset + channel_length + channel_right_offset
                
                if (False):
                    
                    print()
                    print("s2 window all chan:   " + str(event_s2_length))
                    print("channel length:       " + str(channel_length))
                    print("channel left offset:  " + str(channel_left_offset))
                    print("channel right offset: " + str(channel_right_offset))
                    print("event:       " + str(event_num)      )
                    print("left chan:   " + str(channel_left)       )
                    print("left evt:    " + str(event_s2_left)  )
                    print("right chan:  " + str(channel_right)      )
                    print("right evt:   " + str(event_s2_right) )
                    print("length chan: " + str(channel_length)     )
                    print("length evt:  " + str(event_s2_length))
            
                assert(channel_left    >= event_s2_left )
                assert(channel_right   <= event_s2_right)
                assert(event_s2_length == channel_left_offset + channel_length + channel_right_offset  )

                arr_channel[channel_left_offset : channel_left_offset + channel_length] = channel_raw_data  
                
                assert( abs(channel_integral - np.sum(arr_channel)) < 1e-4 )
    
            
            #------------------------------------------------------------------
            # Pad to the widest S2 over all events
            #------------------------------------------------------------------
                
            arr_channel_padded                       = np.zeros(s2_window_max)
            arr_channel_padded[0 : arr_channel.size] = arr_channel
            event_train_data[:, iChannel]            = arr_channel_padded # timeseries for iChannel

                  
            #------------------------------------------------------------------
            # End loop over channels
            #------------------------------------------------------------------
            
            continue
    
        
        #----------------------------------------------------------------------
        #----------------------------------------------------------------------
        
        train_data [iEvent, :, :] = event_train_data
        event_train_truth[0]      = x_true
        event_train_truth[1]      = y_true
        train_truth[iEvent, :]    = event_train_truth

        
        #----------------------------------------------------------------------
        # Sanity
        #----------------------------------------------------------------------
        
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
        
        
        #----------------------------------------------------------------------
        # End loop over events
        #----------------------------------------------------------------------
    
        continue

        
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------

    print(str(nEmpty) + " empty events")

    return train_data, train_truth


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def getEventsDataFrame(input_path):

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------

    events_df = pd.read_pickle(input_path)
    events_df = events_df[events_df['intr_count'] > 0] 

    arr_s2_right_minus_left = (events_df['event_s2_right']- events_df['event_s2_left']).as_matrix()
    arr_s2_length           =  events_df['event_s2_length'].as_matrix()  
    
    
    #--------------------------------------------------------------------------
    # Sanity
    #--------------------------------------------------------------------------
    
    s2_window_max = np.amax(arr_s2_right_minus_left) + 1
    s2_length_max = np.amax(arr_s2_length)
    
    assert(s2_window_max == s2_length_max)
    
    
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    return events_df, s2_window_max
    