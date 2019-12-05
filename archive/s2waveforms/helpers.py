
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

import numpy as np
import pandas as pd


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def waveformIndexRankedByIntegral(df_pmts, low, high):

    arr_sum = np.zeros(0)
    arr_idx = np.zeros(0)

    for i in range(0, len(df_pmts.columns)):
           
        arr_pmt  = df_pmts.iloc[:, i].as_matrix()
        integral = np.sum(arr_pmt[low:high])
        arr_sum  = np.append(arr_sum, integral)
        arr_idx  = np.append(arr_idx, int(i))

        continue

    df = pd.DataFrame(data={'pmt_s2_integral': arr_sum, 'pmt_index': arr_idx})
    df = df.sort_values(['pmt_s2_integral'], ascending=False)
    df = df.reset_index(drop=True)

    return df


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def sumWaveformsOverPMTs(df_pmts, idxs_pmts, low, high):

    nSamples   = len(df_pmts.index)
    arr_s2_sum = np.zeros(nSamples)
    
    for i in range(0, idxs_pmts.size):
           
        idx        = int(idxs_pmts[i])
        arr_pmt    = df_pmts.iloc[:, idx]
        s2_sum     = np.sum(arr_pmt[low:high])
        arr_s2_sum = np.add(arr_s2_sum, arr_pmt)

        continue

    return arr_s2_sum
