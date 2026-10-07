### UPDATED-VECTORIZED BETA

#!/usr/bin/env python
import numpy as np
import subprocess
import pickle
import copy
import random
from tools.util import int2str,find_range

import Namelist as gv

seed = gv.random_seed
rng = np.random.default_rng(seed=seed)

local_random_state = gv.local_random

def get_E1(bt):
    """
    get errors and residual of errors from predictand bt
    return:
      errors(E1), 
      [v0E],
      ranges of v0 (cat1)
    """

    NS = np.where((bt['StormYear']>=1981)&(bt['StormYear']<=2012))[0]

    data  = bt['errors'][:, NS].values          # shape (T, N)
    data2 = bt['StormMwspd'][:, NS].values      # shape (T, N)

    data1 = data.ravel(order = 'F')
    data4 = data2.ravel(order = 'F')

    ## mask out NaNs in data1
    nan_mask = ~np.isnan(data1)

    E1  = data1[nan_mask]
    E0  = data1[nan_mask]      
    v0E = data4[nan_mask]

    E2 = E0 * E1 * v0E
    mask = (E2 == E2)
    E0 = E0[mask]
    E1 = E1[mask]
    v0E = v0E[mask]
    c11, c0 = np.polyfit(E0, E1, 1)

    vals = v0E
    vmin = vals.min()
    vmax = vals.max()

    ## threshold grids 
    grid_up = np.arange(vmin, vmax, 10)       # for c1 (min → max)
    grid_dn = np.arange(vmax, vmin, -10)      # for c2 (max → min)

    ### c1 
    ## for every threshold iC in grid_up, compute how many values satisfy v0E < iC
    ## (n_vals,1) < (1,n_thresholds)
    counts_lt = (vals[:, None] < grid_up[None, :]).sum(axis=0)

    ## Pick first threshold where count > 50
    idx = np.argmax(counts_lt > 50) if np.any(counts_lt > 50) else len(grid_up)-1
    c1 = grid_up[idx]

    ### c2 
    # for every threshold iC in grid_dn, compute how many values satisfy v0E > iC
    counts_gt = (vals[:, None] > grid_dn[None, :]).sum(axis=0)

    idx = np.argmax(counts_gt > 50) if np.any(counts_gt > 50) else len(grid_dn)-1
    c2 = grid_dn[idx]

    ## final range
    cat1 = np.arange(c1, c2 + 10, 10)

    return E0,v0E,cat1

def get_mean(meanFile,predictors):
    '''
    calculate mean in X and Y based on predictors
    '''
    hour = range(12,132,12)
    n2 = len(predictors)
    meanX = np.empty([len(hour),n2],dtype=float)+np.float('nan')
    meanY = np.empty([len(hour)],dtype=float)+np.float('nan')
    stdX  = np.empty([len(hour),n2],dtype=float)+np.float('nan')
    stdY  = np.empty([len(hour)],dtype=float)+np.float('nan')
    f = open(meanFile)
    b = f.readlines()
    f.close()
    count1 = 0
    for ih in hour:
        lm = b[ih/12].rstrip().rsplit()
        meanX[count1,:] = lm[3:3+n2]
        meanY[count1]  = lm[1]
        stdX[count1,:] = lm[3+n2::]
        stdY[count1]   = lm[2]
        count1 += 1
    return meanX,meanY,stdX,stdY


def findError(errort, v0E, v0_array, cat1, storm_ids, time_step, ensemble_member):
    '''
    Calculate error for multiple storms
    '''
    nStorms = v0_array.shape[0]
    result = np.zeros(nStorms)
    
    for iS in range(nStorms):
        ## LOCAL ERROR
        storm_id = storm_ids[iS]

        v0 = v0_array[iS]
        if np.isnan(v0):
            result[iS] = 0
        else:
            if v0 <= cat1[0]:
                mask = v0E < cat1[1]
            elif v0 >= cat1[-1]:
                mask = v0E >= cat1[-1]
            else:
                # find using searchsorted (faster than find_range)
                j0 = np.searchsorted(cat1, v0, side='right') - 1
                mask = (v0E >= cat1[j0]) & (v0E < cat1[j0+1])
            
            error1 = errort[mask]
            
            ## *** NON LOCAL ERROR ***
            if local_random_state == False: 
                result[iS] = rng.choice(error1) if len(error1) > 0 else 0

            ## *** LOCAL ERROR ***
            if local_random_state == True: 
                local_seed = seed + ensemble_member * 1000000 + storm_id * 10000 + time_step * 10
                ## debugging
                #random_call_log_vec.append((ensemble_member, storm_id, time_step, local_seed))
                local_rng = np.random.default_rng(seed=local_seed)
                result[iS] = local_rng.choice(error1) if len(error1) > 0 else 0

    return result

