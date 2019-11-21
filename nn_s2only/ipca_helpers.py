
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

import glob
import numpy as np
import os
import psutil
import scipy.stats
import time


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def downsample_arr3d(x3d, downsample_factor):

    assert(len(x3d.shape) == 3)
    assert(x3d.shape[2] % downsample_factor == 0)

    return np.sum(x3d.reshape(x3d.shape[0], x3d.shape[1], -1, downsample_factor), axis=3)


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def looper(func, lst_dir_files, maxRows, n_events_per_batch):
    
    n_batches = int(maxRows/n_events_per_batch)
    
    print("\nbatches per file: {0}\n".format(n_batches))
    
    
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
    
    for iFile, fpath in enumerate(lst_dir_files):
    
        #----------------------------------------------------------------------
        #----------------------------------------------------------------------
        
        print()
        print(fpath)
        print("   Loading data...")
        
        sArr  = np.load(fpath)['arr_0']
        #arr3d = sArr[0:maxRows][:]['image']
        j0    = iFile*maxRows
        j1    = j0 + maxRows - 1
        
            
        #----------------------------------------------------------------------
        #----------------------------------------------------------------------
        
        for ibatch in range(n_batches):
           
            #------------------------------------------------------------------
            # These indices are important
            #------------------------------------------------------------------
        
            i0  = int(ibatch*n_events_per_batch)
            i1  = int(i0 + n_events_per_batch - 1)
            tb1 = time.time()
            
            print("   Processing Input Data: Fitting batch {0}/{1}   (events: {2}-{3})".format(
                ibatch+1,
                n_batches,
                i0,
                i1
            ))
            
            
            #------------------------------------------------------------------
            # Here we call a function passed to this function
            #------------------------------------------------------------------
            
            k0  = j0 + ibatch*n_events_per_batch
            k1  = k0 + n_events_per_batch
                
            func(sArr[i0:i1+1][:][:], k0, k1)

               
            #------------------------------------------------------------------
            #------------------------------------------------------------------
            
            tb2    = time.time()
            dtb    = tb2 - tb1
            
            #print("      -> Processing took {0:.2f} min., Mem={1} GB".format(dtb/60, getMemoryGB(proc)))
            
            continue
                  
        
        #----------------------------------------------------------------------
        #----------------------------------------------------------------------
        
        continue
    
        
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
        
    #assert(scaler.n_samples_seen_ == n_dir*events_per_dir)
    
    return


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def mkdirIfEmpty(dirpath):
    
    if (not os.path.exists(dirpath)):
        
        os.mkdir(dirpath)
        print("created directory\n\t{0}".format(dirpath))
        
    else:
        
        lstDir  = os.listdir(dirpath)
        lstDir  = glob.glob(dirpath + '/*')
        
        isEmpty = len(lstDir) > 0
        
        if (not isEmpty):
            print("Empty directory exists:\n\t{0}".format(dirpath))
            
        else:
            for x in lstDir:
                print(x)
            raise Exception("Directory not empty:\n   {0}".format(dirpath))
        
    return


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def getSizeOnDisk(arr, mytype=np.float64):
    
    outfile = '/tmp/tmp_summedwf.npy'
    np.save(outfile.replace('.npy', ''), arr.astype(mytype))
    sz_mb = round(os.path.getsize(outfile) / 1e6, 3)
    
    return sz_mb


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def getLargestIndexWithNonzeroValue(arr, thresh=0, maxIdx=None):

    ymax   = np.amax(arr)
    idx    = arr.size - 1
    thresh = ymax / 20
    
    if (maxIdx is not None):
        idx = maxIdx
    
    while (idx > 0):
        
        el  = arr[idx]
        
        if (el > thresh):
            #print("i={0}, el={1}".format(idx, el))
            break
        
        idx = idx - 1
        
        continue
    
    return idx


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def meanSumOfSquaredDifferences(arr1d_x, arr1d_y):

    se = sumOfSquaredDifferences(arr1d_x, arr1d_y)
    
    return se / arr1d_x.size


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def sumOfSquaredDifferences(arr1d_x, arr1d_y):

    assert(len(arr1d_x.shape) == 1)
    assert(len(arr1d_y.shape) == 1)
    assert(arr1d_x.size == arr1d_x.size)
    
    arr1d_squared_differences = np.square(arr1d_x - arr1d_y)
    sum_squared_differences   = np.sum(arr1d_squared_differences)
    
    return sum_squared_differences


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def calculate_chi2(arr1d_x, arr1d_y):

    assert(len(arr1d_x.shape) == 1)
    assert(len(arr1d_y.shape) == 1)
    assert(arr1d_x.size == arr1d_x.size)
    
    
    #--------------------------------------------------------------------------
    # Chi2 - may have to downsample for convergence
    #--------------------------------------------------------------------------

    resample_factors = [50]
    chisq            = None
    k                = 0
    
    while (chisq is None):
        
        resample_factor   = resample_factors[k]
        wf_sum_exact_ds2  = downsample(arr1d_x, resample_factor)
        wf_sum_approx_ds2 = downsample(arr1d_y, resample_factor)
        k += 1
        
        #assert(np.isclose(np.sum(wf_sum_approx[i, :, j]), np.sum(wf_sum_approx_ds2)))
    
        chisq = scipy.stats.chisquare(wf_sum_exact_ds2, f_exp=wf_sum_approx_ds2)
        chi2  = chisq.statistic
    
        if(chi2 <= 0):
            print("Warning! Chi2={0}".format(chi2))
    
        continue
    
    assert(chisq is not None)
    
    return chisq


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def downsample(arr1d, factor):

    assert(arr1d.size % factor == 0)
    
    arr1d = arr1d.reshape(factor, int(arr1d.size/factor))
    rval  = np.add.reduce(arr1d, 0)

    return rval



