### UPDATED-VECTORIZED BETA


#!/usr/bin/env python
import numpy as np
import calendar
import random
import time
import sys
import dask.array as da
import os
import gc
import pickle
import copy
import pandas as pd
import xarray as xr


import Namelist as gv
import pandas as pd
import netCDF4 as nc
from netCDF4 import Dataset
from scipy import stats
from tools.util import argminDatetime
from tools.util import int2str,date_interpolation
from scipy.io import loadmat, netcdf_file
from datetime import datetime,timedelta
from dask.diagnostics import ProgressBar

## numba JIT compilation
import numba


from functools import lru_cache

#import calWindCov as calWindCov        ## not pointed at the right place
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

#import calWindCov
#import module_riskModel as mrisk ## not used


## adding random number generator with set seed to allow comparisons and reproducibility
## set seed = None to use computer chaos
seed = gv.random_seed
rng = np.random.default_rng(seed=seed)


def randomSeeding(iy):
    '''
    This function finds random seeding (if TCGIinput = 'random').
    iy: year of current iteration in CHAZ.py
    '''
    climInitLon = []
    climInitLat = []
    climInitDate = []
    a= np.arange(gv.lldmask.size).reshape(gv.lldmask.shape[0],gv.lldmask.shape[1])
    llon1,llat1 = np.meshgrid(gv.llon,gv.llat)
    dummy = np.random.choice(a[(gv.lldmask==0)&(np.abs(llat1)<=65)&(np.abs(llat1)>=3)],gv.seedN)
    y = dummy/a.shape[1]
    x = np.mod(dummy,a.shape[1])
    yeardays = np.array([calendar.monthrange(iy,im)[1] for im in range(1,13)]).sum()
    a = np.arange(0,yeardays,1)
    days = np.random.choice(a,gv.seedN)
    date = pd.date_range(str(iy)+'-01-01', periods=365+calendar.isleap(iy)*1)[days]
    for idd in range(date.size):
        x1 = np.arange(gv.llon[x[idd]]-1,gv.llon[x[idd]]+1.01,0.01)
        y1 = np.arange(gv.llat[y[idd]]-1,gv.llat[y[idd]]+1.01,0.01)
        xx = np.random.choice(x1,1)
        yy = np.random.choice(y1,1)
        climInitDate.append(datetime.combine(date[idd].date(), datetime.min.time()))
        climInitLon.append(xx)
        climInitLat.append(yy)
    climInitDate = np.array(climInitDate).ravel()
    climInitLon = np.array(climInitLon).ravel()
    climInitLat = np.array(climInitLat).ravel()

    return climInitDate, climInitLon,climInitLat

def TCgiSeeding (gxlon, gxlat, gi, climInitLon, climInitLat, climInitDate, ratio, iy):
    '''
    This is a subroutine for calculating tropical cyclone genesis locations based on the
    genesis index (gi).
    
    ---- Parameters -------------------------------------- 
    gxlon : 2D array (nlat x nlon)
        mesh-grid of longitudes

    gxlat : 2D array (nlat x nlon)
        mesh-grid of latitudes
    
    gi : 3D array (month x nlat x nlon)
        Monthly tropical cyclone genisis index (gi). 
        "gi is tropical cyclone genesis index. We use nansum(gi) to for
        total number of seeds, and then randomly select location using gi
        as genesis probability."
            [higher values == more likely genisis?] TODO 
            [spatial probability distribution of cyclone 
            formation for each month.]
    
    climInitLon, climInitLat, climInitDate: lists
        Lists storing storing previous seeds information (lon, lat, and date)

    ratio: float
        how many more seeds we should give [create?], based on the survival rate
        [Scaling factor determining how many seeds to create relative to the total
        genesis index sum (accounts for survival rate).]

    iy: int
        target year for seeding

    ---- Returns ---------------------------------------- 
    None (modifies the climInitLon, climInitLat, and climInitDate lists in place)

    '''

    # loop through year (iy) by month (im) 
    for im in range(12):
        xk, pk = np.arange(gi[im,:,:].size), gi[im,:,:].ravel()

        
        ## remove NaNs
        dummy = xk*pk
        xk, pk = xk[dummy==dummy], pk[dummy==dummy] # (dummy==dummy is False for NaNs)
        # Build discrete probability distribution (custm) using gi values as weights
        # xk = possible grid indices, pk/pk.sum() = normalized probabilities
        custm = stats.rv_discrete(name='custm', values=(xk, pk/pk.sum()), seed = rng)

        ## randomly select grid point from custm 
        ## number of points selected [np.rint(pk.sum()*ratio)]
            # proportional to sum of gensis probability * ratio
        r = custm.rvs(size=np.int_(np.rint(pk.sum()*ratio)))

        ## map selected grid points into lon/lat coordinated
        iix = gxlon.ravel()[r.ravel()]
        iiy = gxlat.ravel()[r.ravel()]
        
        ## randomly select day in month to assign each seed 
        ## update to use set rng
        iday = rng.choice(np.arange(calendar.monthrange(iy,im+1)[1]), size=np.int_(np.rint(pk.sum()*ratio)), replace=True)

        # if there are days assigned a seed in the month:
        if iday.size>0:
            for id in range(iday.size):

                ## this is occasionally generating empty arrays
                x1 = np.arange(iix[id]-1,iix[id]+1.01, 0.01)
                y1 = np.arange(iiy[id]-1,iiy[id]+1.01, 0.01)
               
                ## update to use set rng
                xx = rng.choice(x1, 1)
                yy = rng.choice(y1, 1)

                # if gv.debugging: print(xx)
                # if gv.debugging: print(yy)

                # append generated seed information to appropriate lists
                climInitDate.append(datetime(iy,im+1,iday[id]+1,0,0))
                climInitLon.append(xx)
                climInitLat.append(yy)

    return climInitLon, climInitLat, climInitDate

## updated
def calF(nday):
    '''
    This function finds the fourier function.
    nday: number of days
    '''
    dt = 1.0*60*60         # 1 hr in seconds
    T = np.float64(15)     # 15-day period of the fourier series
    N = 15                 # number of sine waves
    nt = np.arange(0, nday*60*60*24, dt)

    ## define iN and itt (shape/data holders)
    iN = np.arange(1, N+1, dtype = float)              ## shape (N,)
    itt = np.arange(nt.size, dtype = float)            ## shape (nt,)
    #X = np.random.uniform(0, 1, (N, 4))               ## random phases for 4 columns

    # use random number generator with set seed
    X = rng.uniform(0, 1, (4, N)).T                     ## transposed to match original shape 

    ## TODO: possible speedup w/ numba accelleration
    ##  vectorized equation
    F = (np.sqrt(2.0 / np.sum(iN**-3)) *
            np.sum(
                (iN**(-3/2))[:, None, None] * 
                np.sin(2*np.pi * ((iN[:, None, None] * itt[None, :, None]) / (24*T) + X[:, None, :])),
                axis=0
            )
    )

    return F


## adding numba acceleration 
@numba.jit(nopython=True)
def _getLonLatfromDistance(lonInit,latInit,dx,dy):
    '''
    This function calculates the latitude from the distance.
    lonInit: initial longitude
    latInit: initial latitude
    dx: x differential
    dy: y differential
    '''
    er = 6371000 #km
    londis = 2*np.pi*er*np.cos(latInit/180*np.pi)/360.
    lon2 = lonInit+dx/londis
    latdis = 2*np.pi*er/360.
    lat2 = latInit+dy/latdis
    return lon2,lat2

## added
@numba.jit(nopython = True)
def _getTrackPrediction(u250, v250, u850, v850, dt, fstLat, uBeta, vBeta):
    fstLat_rad = fstLat*np.pi/180.
    alpha = 0.8
    uTrack = alpha * u850 + (1.-alpha)*u250 + uBeta
    vTrack = alpha * v850 + (1.-alpha)*v250 + vBeta * np.sign(np.sin(fstLat_rad))
    dx = uTrack*dt
    dy = vTrack*dt
    return dx, dy

## updated
def getTrackPrediction(u250,v250,u850,v850,dt,fstLon,fstLat,fstDate):
    '''
    This function predicts the track.
    u250: zonal wind time series at 250 hPA.
    v250: meridional wind time series at 250 hPA.
    u850: zonal wind time series at 850 hPA.
    v850: meridional wind time series at 850 hPA.
    dt: time differential.
    fstLon: longitude 
    fstLat: latitude
    fstDate: date
    '''

    #### modify Beta
    earth_rate = 7.2921150e-5              # mean earth rotation rate in radius per second
    r0 = 6371000                           # mean earth radius in m
    lat0 = np.arange(-90,100,10)
    phi0 = lat0/180.*np.pi                 # original latitude in radian (ex. at 15 degree N)
    beta0 = 2.0*earth_rate*np.cos(phi0)/r0 # per second per m
    beta0 = beta0/beta0[10]
    ratio = np.interp(fstLat,np.arange(-90,100,10),beta0)
    uBeta = gv.uBeta*ratio
    vBeta = gv.vBeta*ratio
    ################

    dx, dy = _getTrackPrediction(u250, v250, u850, v850, dt, fstLat, uBeta, vBeta)

    lon2,lat2 = _getLonLatfromDistance(fstLon, fstLat, dx,dy)
    fstLon,fstLat = lon2,lat2
    fstDate += timedelta(seconds=dt)
    #print uBeta, vBeta,fstLon,fstLat

    return fstLon,fstLat,fstDate





### helper functions to load datasets a single time and 
### hold in memory, rather than importing multiple times in loop 
@lru_cache(maxsize=None)
def load_I_MLR(year):
    '''
    TODO: ADD
    '''

    fpath = gv.pre_path

    fname =  fpath+int2str(year,4)+'_'+gv.ENS+'.nc'
    if not gv.quiet: print(f"Lazy loading from disk for year {year}")                            ## check that this only runs once 
    #print(fname)
    ds = xr.open_dataset(fname)
    #out = {k: ds[k].values for k in ['u250','v250','u850','v850']}             ### different variable names
    #out = {k: ds[k].values for k in ['ua2502', 'va2502', 'ua8502', 'va8502']}  ### may need to adjust and/or dynamically rename? 
    
    ## version to easily rename variables 
    key_map = { 
        'ua2502': 'u250',
        'va2502': 'v250',
        'ua8502': 'u850',
        'va8502': 'v850'}

    out = {v: ds[k].values for k, v in key_map.items()}

    ## probably should write this more robustly
    if 'lon' in ds.coords:                
        lon, lat = ds['lon'].values, ds['lat'].values
    elif 'Longitude' in ds.coords:
        lon, lat = ds['Longitude'].values, ds['Latitude'].values
    else:
        print('unrecognized lat/lon coordinate names')

    ds.close()
    return out, lon, lat

@lru_cache(maxsize=None)
def load_A(year, month):
    '''
    TODO: ADD 
    '''
    fpath = gv.pre_path

    #fname = pooch.retrieve(url=f"{path_data}/A_{year:04d}{month:02d}.nc", known_hash=None)
    fname = fpath+'A_'+int2str(year,4)+int2str(month,2)+'.nc'
    
    ## check that this loads only when expected
    #print(f"\n    Loading A from disk for year {year} and month {month}")                       
    ds = xr.open_dataset(fname)
    A = ds['A'].load().values
    ds.close()
    return A

### updated
## possibly room to split out some of the calculations in this 
## and numba accelerate
def bam(iS,block_id=None):
    """
    Beta-Advection Model (BAM)
    Simulates the track of a tropical cyclone.

    Parameters:
    iS (array): Index of the initial condition for the simulation.
    block_id (optional): Dask block identifier.

    Returns:
    ##int: Index of the initial condition.
    """
    # [1] Generate F from Emanuel et. al. (2006). It is a Fourier series where the individual wave components have a random phase.
    # neighbors=((0,1),(0,-1),(1,0),(-1,0),(1,1),(-1,1),(1,-1),(-1,-1),(0,2),(0,-2),(2,0),(-2,0)) 
    # missing_value = 1e+20

    dt = 1.0*60*60
    T = np.float64(15)      # 15-day period of the fourier series (i think)
    F = calF(15)            # computer fourier series 
    #print(F)
    nt = np.arange(0,T*60*60*24,dt)
    b = np.int_(iS.mean(keepdims=True)[0])
    #b = iS
    #print(b, 'b')

    #initialize start position/date
    fstDate = climInitDate[b]
    fstLon = climInitLon[b]
    fstLat = climInitLat[b]
    fstlon[0,b] = fstLon
    fstlat[0,b] = fstLat
    fstldmask[0,b] = 0

    endhours = fstlon.shape[0]-1
    endDate = climInitDate[b] + timedelta(hours = endhours)
    count,year0,month0,day0 = 1,0,0,0

    ### using 0.75 ldmask ###
    ldxxlong,ldxxlat = np.meshgrid(gv.ldlon,gv.ldlat)
    #########################

    ## Update
    ## used cache loading function
    A_matrix = load_A(fstDate.year, fstDate.month)
    month0 = fstDate.month
 
    #print(endDate)
    #while fstDate < endDate and fstDate.year==yearTC:          ## TODO: from tutorial optimization unclear if needed
    while fstDate < endDate:                                    ## Original/most correct
    #while fstDate < endDate and fstDate.year == endDate.year:   ## this keeps it from loading the next year, but not sure if i'm breaking things only running one year
        if fstDate.year != year0:
            ## load dataset cached once and reference 
            year_data, xlong, xlat = load_I_MLR(fstDate.year)
            u250m = year_data['u250']
            u850m = year_data['u850']
            v250m = year_data['v250']
            v850m = year_data['v850']
            
            xxlong,xxlat = np.meshgrid(xlong,xlat)
            #ds.close()
            year0 = fstDate.year

        if fstDate.day != day0:
            #print fstDate.day,day0
            u250m2d = date_interpolation(fstDate,u250m)
            v250m2d = date_interpolation(fstDate,v250m)
            u850m2d = date_interpolation(fstDate,u850m)
            v850m2d = date_interpolation(fstDate,v850m)
            day0=fstDate.day

        ## updated to chache loading -- this saves a lot of time 
        ## because dataset was fully loaded into memory for each date
        if (fstDate.month != month0):
            A_matrix = load_A(fstDate.year, fstDate.month)
            # FileAName = pooch.retrieve(url=f"{path_data}/A_{fstDate.year:04d}{fstDate.month:02d}.nc",  known_hash=None)
            # dsA = xr.open_dataset(FileAName)
            # A_matrix = dsA.variables['A'].load()
            # dsA.close()
            month0 = fstDate.month
        A = A_matrix[:,fstDate.day-1,:,:]
        day0 = fstDate.day

        ## unchanged
        distance = np.sqrt((fstLon-xxlong)**2+(fstLat-xxlat)**2)
        iy,ix = np.unravel_index(np.argmin(distance),distance.shape)
        iy1,ix1 = np.max([iy-2,0]),np.max([ix-2,0])
        iy2,ix2 = np.min([iy+2,distance.shape[0]]),np.min([ix+2,distance.shape[1]])

        iit = np.mod(count,nt.shape[0])
        u250 = u250m2d[iy1:iy2+1,ix1:ix2+1]+A[0,iy1:iy2+1,ix1:ix2+1]*F[iit,0]
        v250 = v250m2d[iy1:iy2+1,ix1:ix2+1]+A[1,iy1:iy2+1,ix1:ix2+1]*F[iit,0]+A[2,iy1:iy2+1,ix1:ix2+1]*F[iit,1]
        u850 = u850m2d[iy1:iy2+1,ix1:ix2+1]+A[3,iy1:iy2+1,ix1:ix2+1]*F[iit,0]+\
                         A[4,iy1:iy2+1,ix1:ix2+1]*F[iit,1]+A[5,iy1:iy2+1,ix1:ix2+1]*F[iit,2]
        v850 = v850m2d[iy1:iy2+1,ix1:ix2+1]+A[6,iy1:iy2+1,ix1:ix2+1]*F[iit,0]+\
                         A[7,iy1:iy2+1,ix1:ix2+1]*F[iit,1]+A[8,iy1:iy2+1,ix1:ix2+1]*F[iit,2]+\
                         A[9,iy1:iy2+1,ix1:ix2+1]*F[iit,3]


        u250 = np.nanmean(u250)
        u850 = np.nanmean(u850)
        v250 = np.nanmean(v250)
        v850 = np.nanmean(v850)

        fstLon, fstLat, fstDate = getTrackPrediction(u250,v250,u850,v850,dt,fstLon,fstLat,fstDate)
        #print fstDate,year0,day0
        if ((fstLon<0.0) or (fstLon>360) or (fstLat<-60) or (fstLat>60)):
            #print b, 'break for going to the space'
            break
        fstlon[count,b] = fstLon
        fstlat[count,b] = fstLat
        
        ### recalculating the ix,iy landmask for .75º ####
        distance = np.sqrt((fstLon-ldxxlong)**2+(fstLat-ldxxlat)**2)
        iy,ix = np.unravel_index(np.argmin(distance),distance.shape)
        iy1,ix1 = np.max([iy-2,0]),np.max([ix-2,0])
        iy2,ix2 = np.min([iy+2,distance.shape[0]]),np.min([ix+2,distance.shape[1]])
        fstldmask[count,b] = np.rint(np.nanmean(gv.ldldmask[iy1:iy2+1,ix1:ix2+1]))
        ##################################################
        #fstldmask[count,b] = np.rint(np.nanmean(gv.ldmask[iy1:iy2+1,ix1:ix2+1]))
        
        count += 1

    return b


# ### from wdir-Landmask
# def bam(iS,block_id=None):
#     #print iS
#     neighbors=((0,1),(0,-1),(1,0),(-1,0),(1,1),(-1,1),(1,-1),(-1,-1),(0,2),(0,-2),(2,0),(-2,0))
#     dt = 1.0*60*60
#     T = 15
#     N = 15
#     F = calF(15)
#     nt = np.arange(0,T*60*60*24,dt)
#     b = np.int_(iS.mean(keepdims=True)[0])
#     #b = iS
#     fstDate = climInitDate[b]
#     fstLon = climInitLon[b]
#     fstLat = climInitLat[b]
#     fstlon[0,b] = fstLon
#     fstlat[0,b] = fstLat
#     fstldmask[0,b] = 0
#     endhours = fstlon.shape[0]-1
#     endDate = climInitDate[b] + timedelta(hours = endhours)
#     count,year0,month0,day0 = 1,0,0,0
#     fpath = './'

#     ### using 0.75 ldmask ###
#     ldxxlong,ldxxlat = np.meshgrid(gv.ldlon,gv.ldlat)
#     ######

#     FileAName =fpath+'A_'+int2str(fstDate.year,4)+int2str(fstDate.month,2)+'.nc'
#     dsA = xr.open_dataset(FileAName)
#     A_matrix = dsA.variables['A'].load()
#     dsA.close()
#     month0 = fstDate.month

#     while fstDate < endDate:
#         if (fstDate.year != year0) :
#             fileName = fpath+int2str(fstDate.year,4)+'_'+gv.ENS+'.nc'
#             #print fstDate.year,year0
#             if os.path.isfile(fileName):
#                 nc = Dataset(fileName,'r',format='NETCDF3_CLASSIC')
#                 u250m = nc.variables['ua2502'][:]
#                 u850m = nc.variables['ua8502'][:]
#                 v250m = nc.variables['va2502'][:]
#                 v850m = nc.variables['va8502'][:]
#                 u850m = calWindCov.fillinNaN(u850m,neighbors)
#                 v850m = calWindCov.fillinNaN(v850m,neighbors)
#                 xlong = nc.variables['Longitude'][:]
#                 xlat = nc.variables['Latitude'][:]
#                 xxlong,xxlat = np.meshgrid(xlong,xlat)
#                 nc.close()
#                 year0 = fstDate.year
#             else:
#                 print('no'+fileName)
#                 break
#         if fstDate.day != day0:
#             #print fstDate.day,day0 
#             u250m2d = date_interpolation(fstDate,u250m)
#             v250m2d = date_interpolation(fstDate,v250m)
#             u850m2d = date_interpolation(fstDate,u850m)
#             v850m2d = date_interpolation(fstDate,v850m)
#             day0=fstDate.day

#             #FileAName =\
#             #fpath+'A_'+int2str(fstDate.year,4)+int2str(fstDate.month,2)+'.nc'
#             #ncA = Dataset(FileAName,'r',format='NETCDF3_CLASSIC')
#             #A = ncA.variables['A'][:,fstDate.day-1,:,:]
#             #ncA.close()

#             if (fstDate.month != month0):
#                 FileAName = fpath+'A_'+int2str(fstDate.year,4)+int2str(fstDate.month,2)+'.nc'
#                 dsA = xr.open_dataset(FileAName)
#                 A_matrix = dsA.variables['A'].load()
#                 dsA.close()
#                 month0 = fstDate.month
#             A = A_matrix[:,fstDate.day-1,:,:]
#             day0 = fstDate.day



#             distance = np.sqrt((fstLon-xxlong)**2+(fstLat-xxlat)**2)
#             iy,ix = np.unravel_index(np.argmin(distance),distance.shape)
#             iy1,ix1 = np.max([iy-2,0]),np.max([ix-2,0])
#             iy2,ix2 = np.min([iy+2,distance.shape[0]]),np.min([ix+2,distance.shape[1]])

#             iit = np.mod(count,nt.shape[0])
#             u250 = u250m2d[iy1:iy2+1,ix1:ix2+1]+A[0,iy1:iy2+1,ix1:ix2+1]*F[iit,0]
#             v250 = v250m2d[iy1:iy2+1,ix1:ix2+1]+A[1,iy1:iy2+1,ix1:ix2+1]*F[iit,0]+A[2,iy1:iy2+1,ix1:ix2+1]*F[iit,1]
#             u850 = u850m2d[iy1:iy2+1,ix1:ix2+1]+A[3,iy1:iy2+1,ix1:ix2+1]*F[iit,0]+\
#                                             A[4,iy1:iy2+1,ix1:ix2+1]*F[iit,1]+A[5,iy1:iy2+1,ix1:ix2+1]*F[iit,2]
#             v850 = v850m2d[iy1:iy2+1,ix1:ix2+1]+A[6,iy1:iy2+1,ix1:ix2+1]*F[iit,0]+\
#                                             A[7,iy1:iy2+1,ix1:ix2+1]*F[iit,1]+A[8,iy1:iy2+1,ix1:ix2+1]*F[iit,2]+\
#                                             A[9,iy1:iy2+1,ix1:ix2+1]*F[iit,3]

#             #try: 
#             #   mytest = np.min(u850[~u850.mask])
#             #except ValueError: 
#             #   print 'mountainous'
#             #   break;

#             u250 = np.nanmean(u250)
#             u850 = np.nanmean(u850)
#             v250 = np.nanmean(v250)
#             v850 = np.nanmean(v850)

#             fstLon, fstLat, fstDate = getTrackPrediction(u250,v250,u850,v850,dt,fstLon,fstLat,fstDate)
#             #print fstDate,year0,day0
#             if ((fstLon<0.0) or (fstLon>360) or (fstLat<-60) or (fstLat>60)):
#                 #print b, 'break for going to the space'
#                 break
#             fstlon[count,b] = fstLon
#             fstlat[count,b] = fstLat
#             ### recalculating the ix,iy landmask ####
#             distance = np.sqrt((fstLon-ldxxlong)**2+(fstLat-ldxxlat)**2)
#             iy,ix = np.unravel_index(np.argmin(distance),distance.shape)
#             iy1,ix1 = np.max([iy-2,0]),np.max([ix-2,0])
#             iy2,ix2 = np.min([iy+2,distance.shape[0]]),np.min([ix+2,distance.shape[1]])
#             fstldmask[count,b] = np.rint(np.nanmean(gv.ldldmask[iy1:iy2+1,ix1:ix2+1]))
#             ###########################################
#             del u250,u850,v250,v850
#             count += 1
#     return b


def get_landmask(filename):
    '''
    This function reads landmask.nc. 
    lon: longitude, 1D
    lat: latitude, 1D
    landmask: 2D
  
    '''
    f = netcdf_file(filename)
    lon = f.variables['lon'][:]
    lat = f.variables['lat'][:]
    landmask = f.variables['landmask'][:,:]
    f.close()

    return lon, lat, landmask

def removeland(iS):
    '''
    This function removes land from the landmask.
    '''
    #print iS
    b = np.int_(iS.mean(keepdims=True))
    iT3 = -1
    if b<fstldmask.shape[1]:
        a = np.argwhere(fstldmask[:,b]==3).ravel()
        if a.size:
           if a.size>3:
              iT3 = a[0]+2
              fstlon[iT3:,b]=np.nan
              fstlat[iT3:,b]=np.nan
    return iT3

## get predictors:
class fst2bt(object):
    '''
    This converts data format from fst to be format of bt object.
    '''
    def __init__(self,data):
        self.StormId = np.arange(0,data['lon'].shape[1],1)
        self.StormYear = []
        self.StormInitMonth = []
        
        for iS in self.StormId:
            if data['Time'][0,iS] is not None:
               self.StormYear.append(data['Time'][0,iS].year)
               self.StormInitMonth.append(data['Time'][0,iS].month)
            else:
               self.StormYear.append(1800)
               self.StormInitMonth.append(1)
        
        self.StormYear = np.array(self.StormYear)
        self.StormInitMonth = np.array(self.StormInitMonth)
        
        for iS in range(data['Time'].shape[1]):
            data['Time'][:,iS] = np.array([datetime(1800,1,1,0) if v is None else v for v in data['Time'][:,iS]])
        
        data['Time'][data['lon']!=data['lon']] = datetime(1800,1,1,0)
        data['lon'][data['Time']==datetime(1800,1,1,0)] = np.float64('Nan')
        
        self.StormLon = data['lon']
        self.StormLat = data['lat']
        self.Time = data['Time']
        self.PIwspd = np.empty(data['lon'].shape,dtype=float)*np.float64('Nan')
        self.PIslp = np.empty(data['lon'].shape,dtype=float)*np.float64('Nan')
        self.PIwspdMean = np.empty(data['lon'].shape,dtype=float)*np.float64('Nan')
        self.dPIwspd = np.empty(data['lon'].shape,dtype=float)*np.float64('Nan')
        self.PIslpMean = np.empty(data['lon'].shape,dtype=float)*np.float64('Nan')
        self.UShearMean = np.empty(data['lon'].shape,dtype=float)*np.float64('Nan')
        self.UShear = np.empty(data['lon'].shape,dtype=float)*np.float64('Nan')
        self.VShearMean = np.empty(data['lon'].shape,dtype=float)*np.float64('Nan')
        self.VShear = np.empty(data['lon'].shape,dtype=float)*np.float64('Nan')
        self.div200Mean = np.empty(data['lon'].shape,dtype=float)*np.float64('Nan')
        self.div200 = np.empty(data['lon'].shape,dtype=float)*np.float64('Nan')
        self.T200Mean = np.empty(data['lon'].shape,dtype=float)*np.float64('Nan')
        self.T200 = np.empty(data['lon'].shape,dtype=float)*np.float64('Nan')
        self.rh500_300Mean = np.empty(data['lon'].shape,dtype=float)*np.float64('Nan')
        self.rh500_300 = np.empty(data['lon'].shape,dtype=float)*np.float64('Nan')
        self.rhMean = np.empty(data['lon'].shape,dtype=float)*np.float64('Nan')
        self.rh = np.empty(data['lon'].shape,dtype=float)*np.float64('Nan')
        self.T100Mean = np.empty(data['lon'].shape,dtype=float)*np.float64('Nan')
        self.T100 = np.empty(data['lon'].shape,dtype=float)*np.float64('Nan')
        self.dThetaEMean = np.empty(data['lon'].shape,dtype=float)*np.float64('Nan')
        self.dThetaE = np.empty(data['lon'].shape,dtype=float)*np.float64('Nan')
        self.dThetaEs = np.empty(data['lon'].shape,dtype=float)*np.float64('Nan')
        self.dThetaEsMean = np.empty(data['lon'].shape,dtype=float)*np.float64('Nan')
        self.landmask = data['ldmask']
        self.landmaskMean = data['ldmask']
        self.trSpeed = np.zeros(data['lon'].shape)*np.float64('Nan')
        self.trDir = np.zeros(data['lon'].shape)*np.float64('Nan')
        self.dVdt = np.zeros(data['lon'].shape)
        

def func_first(x):
    '''
    This function returns the first non NA/Null value.
    '''
    if x.first_valid_index() is None:
        return None
    else:
        return x.first_valid_index()

def func_last(x):
    '''
    '''
    if x.last_valid_index() is None:
        return None
    else:
        return x.last_valid_index()


def get_predictors(iiS,block_id=None):
    '''
    This function returns an object containing the predictors.
    '''
    iS = np.int_(iiS.mean(keepdims=True))[0]
    #predictors=['StormMwspd','dVdt','trSpeed','dPIwspd','SHRD','rhMean','dPIwspd2','dPIwspd3','dVdt2','landmaskMean']
    
    #if bt.StormYear[iS]>=1980:
    fpath = gv.pre_path
    fileName = fpath+int2str(bt.StormYear[iS],4)+'_'+gv.ENS+'.nc'
    ### easy speedup to more compressible updated format
    #nc = Dataset(fileName,'r',format='NETCDF3_CLASSIC')
    nc = Dataset(fileName, 'r', format = 'NETCDF4')
    xlong = nc.variables['Longitude'][:]
    xlat = nc.variables['Latitude'][:]
    PIVmax = nc.variables['PI2'][:,:,:]*1.94384449 #m/s - kt
    u250 = nc.variables['ua2502'][:]
    u850 = nc.variables['ua8502'][:]
    v250 = nc.variables['va2502'][:]
    v850 = nc.variables['va8502'][:]
    u = u250-u850
    v = v250-v850
    meanrh = nc.variables['hur2'][:]
    xxlong,xxlat = np.meshgrid(xlong,xlat)

    del xlong, xlat
    nc.close()
	
    ## updated - moved outside of for loop
    ## 2º
    er = 6371.0 #km
    londis = 2*np.pi*er*np.cos(xxlat/180*np.pi)/360
    distance = np.empty(xxlong.shape,dtype=float)     

    ### TODO: Double check this is what I want to be doing here still
    ### using 0.75 ldmask ###
    ldxxlong,ldxxlat = np.meshgrid(gv.ldlon,gv.ldlat)
    ######

    ## .75º
    distance75 = np.empty(ldxxlong.shape,dtype=float)
    londis75 = 2*np.pi*er*np.cos(ldxxlat/180*np.pi)/360

    for it in range(0,bt.StormLon[:,iS].shape[0],1):	
        if bt.Time[it,iS] != datetime(1800,1,1,0):
            dx = londis*(xxlong - bt.StormLon[it,iS])
            dy = 110 * (xxlat - bt.StormLat[it,iS])
            distance = np.sqrt(dx*dx+dy*dy)
            (j0,i0) = np.unravel_index(np.argmin(distance),distance.shape)

            var,radius1,radius2 =  date_interpolation(bt.Time[it,iS],PIVmax),0,500
            bt.PIwspdMean[it,iS] = np.mean(var[(distance<=radius2) & (distance>=radius1) & (var==var)])
            bt.PIwspd[it,iS] = var[j0,i0]

            var,radius1,radius2 =  date_interpolation(bt.Time[it,iS],u),200,800
            bt.UShearMean[it,iS] = np.mean(var[(distance<=radius2) & (distance>=radius1) & (var==var)])
            bt.UShear[it,iS] = var[j0,i0]

            var,radius1,radius2 =  date_interpolation(bt.Time[it,iS],v),200,800
            bt.VShearMean[it,iS] = np.mean(var[(distance<=radius2) & (distance>=radius1) & (var==var)])
            bt.VShear[it,iS] = var[j0,i0]

            var,radius1,radius2 =  date_interpolation(bt.Time[it,iS],meanrh),200,800
            bt.rhMean[it,iS] = np.mean(var[(distance<=radius2) & (distance>=radius1) & (var==var)])
            bt.rh[it,iS] = var[j0,i0]

            #var,radius1,radius2 = copy.copy(gv.ldmask),0,300
            ## update for .75º landmask
            var,radius1,radius2 = copy.copy(gv.ldldmask),0,300
            var[var==0] = -1.
            var[var==3] = 0.0

            ### recalculating the ix,iy landmask ####  
            # distance = np.empty(ldxxlong.shape,dtype=float)
            # londis = 2*np.pi*er*np.cos(ldxxlat/180*np.pi)/360
            dx = londis75*(ldxxlong - bt.StormLon[it,iS])  
            dy = 110 * (ldxxlat - bt.StormLat[it,iS])  
            distance75 = np.sqrt(dx*dx+dy*dy)  
            (j0,i0) = np.unravel_index(np.argmin(distance75),distance75.shape)
            ###########################################  

            bt.landmaskMean[it,iS] = np.mean(var[(distance75<=radius2) & (distance75>=radius1) & (var==var)])
            bt.landmask[it,iS] = var[j0,i0] 

    return iS


### updated
def getStormTranslation(lon,lat,time):
    '''
    This function find the storm speed and direction.
    lon: longitude
    lat: latitude
    time: time
    sdir: (ndarray) storm direction
    speed: (ndarray) speed
    '''

    er = 6371.0 #km
    timeInt = []
    lonInt = []
    latInt = []

    for iN in range(time.shape[0]-1):
        timeInt.append(time[iN])
        lonInt.append(lon[iN])
        latInt.append(lat[iN])
        delt = 6
        inv = 1./np.float64(delt)
        for iM in range(1, delt, 1):
            timeInt.append(time[iN]+timedelta(hours=iM))
            lonInt.append((1.-iM*inv)*lon[iN]+iM*inv*lon[iN+1])
            latInt.append((1.-iM*inv)*lat[iN]+iM*inv*lat[iN+1])
    
    ### TODO: don't totally remember where this came from 
    timeInt.append(time[-1])
    lonInt.append(lon[-1])
    latInt.append(lat[-1])
    
    ## convert to arrays for vectorization
    timeInt = np.array(timeInt)
    lonInt = np.array(lonInt)
    latInt = np.array(latInt)
    
    ## vectorized 
    nup = np.array([np.argmin(np.abs(timeInt - (t + timedelta(hours=3)))) for t in time])
    ndn = np.array([np.argmin(np.abs(timeInt - (t - timedelta(hours=3)))) for t in time])
    
    londis = 2*np.pi*er*np.cos(latInt[nup]/180*np.pi)/360
    dx = londis * (lonInt[nup] - lonInt[ndn])
    dy = 110 * (latInt[nup] - latInt[ndn])
    distance = np.sqrt(dx*dx + dy*dy)  # km
    
    sdir = np.arctan2(latInt[nup] - latInt[ndn], lonInt[nup] - lonInt[ndn])
    speed = distance*1000 / (nup - ndn + 1) / 60 / 60  # m/s using actual index difference
    
    return sdir, speed


			
def getSpeedDir(iiS,block_id=None):
    '''
    This function finds the speed and direction of the storm based on 
    '''
    iS = np.int_(iiS.mean(keepdims=True))[0]                            ### need the [0] to work as the vectorization updates
    if (bt.StormLat[:,iS]==bt.StormLat[:,iS]).any():
        it1 = np.argwhere(bt.StormLat[:,iS]==bt.StormLat[:,iS])[0,0]
        it2 = np.argwhere(bt.StormLat[:,iS]==bt.StormLat[:,iS])[-1,0]
        if it2 - it1 >=2:
            bt.trDir[it1:it2,iS],bt.trSpeed[it1:it2,iS]=\
                getStormTranslation(bt.StormLon[it1:it2,iS],\
            bt.StormLat[it1:it2,iS],bt.Time[it1:it2,iS])
    return iS

## not in tutorial, could be a space for speedup? 
def get_seeding_ratio(fpath,iy1,iy2):
    '''
    This function finds the seeding ratio.
    fpath:(str) location of historical simulations/predictions 
    iy1: (int) first year
    iy2: (int) second year
    ##EXP: directory with historic simulations -removed
    ratio: (int) seeding ratio
    '''
    #with open (gv.opath,'r') as f:
    f = gv.opath
    bt_obs = nc.Dataset(f,'r')
    NS = np.argwhere((bt_obs['StormYear'][:]>=iy1)&(bt_obs['StormYear'][:]<=iy2))[:,0].shape[0]
    #totalNS = np.float(NS)/gv.survivalrate
    totalNS = np.float64(NS)/gv.survivalrate
    ratio = 1.0

    for iratio in range(0,1): ## I mad mistake to make it 2 for most of CMIP5 runs, and need to fix that
        ## seeding storms ####
        time1 = time.time()
        climInitLon = []
        climInitLat = []
        climInitDate = []
        for iy in range(iy1,iy2+1,1):
            tcgiFile = fpath+'/'+gv.TCGIinput+'_'+int2str(iy,4)+'.mat'
            gi = np.rollaxis(loadmat(tcgiFile)['TCGI'],2,0)
            #print iy, np.nansum(gi)
            if iy == iy1:
                ## lons in tcgiFile may import as a dtype=uint16
                ## force int16 for later
                xlon =loadmat(tcgiFile)['lon'].astype(np.int16)
                xlat =loadmat(tcgiFile)['lat'].astype(np.int16)
                gxlon,gxlat = np.meshgrid(xlon,xlat)
            climInitLon,climInitLat,climInitDate = \
                TCgiSeeding (gxlon,gxlat,gi,climInitLon,climInitLat,climInitDate,ratio,iy)
        NS = len(climInitLon)
        #print NS, totalNS,ratio
        ratio = np.float64(totalNS)/np.float64(NS)
    return ratio 

## original, with added comments 
def getSeeding(fpath,iy,ratio):
    '''
    This function finds the seeding locations.
    fpath:(str) location of historical simulations/predictions 
    iy: (int) year in CHAZ iteration
    ##EXP: (str) directory with historic simulations --removed
    ratio: (int) seeding ratio
    climInitDate: (ndarray) date
    climInitLon: (ndarray) longitude
    climInitLat: (ndarray) latitude
    '''
    climInitLon = []
    climInitLat = []
    climInitDate = []

    # Load the tropical cyclone genesis index (TCGI)
    tcgiFile = fpath+gv.TCGIinput+'_'+int2str(iy,4)+'.mat'
    gi = np.rollaxis(loadmat(tcgiFile)['TCGI'],2,0)

    ## lons in tcgiFile may import as a dtype=uint16
    ## force int16 for later
    # Load longitude and latitude data
    xlon =loadmat(tcgiFile)['lon'].astype(np.int16)
    xlat =loadmat(tcgiFile)['lat'].astype(np.int16)

    # Create a meshgrid of longitude and latitude
    gxlon,gxlat = np.meshgrid(xlon,xlat)

    # Generate seeds based on the genesis index
    climInitLon,climInitLat,climInitDate = \
       TCgiSeeding (gxlon,gxlat,gi,climInitLon,climInitLat,climInitDate,ratio,iy)
    climInitDate = np.array(climInitDate).ravel()
    climInitLon = np.array(climInitLon).ravel()
    climInitLat = np.array(climInitLat).ravel()
    
    return climInitDate,climInitLon,climInitLat

### updated
def getBam(cDate,cLon,cLat,iy,ichaz):
    '''
    This function creats a beta-advection model object.
    cDate: (ndarray) date
    cLon: (ndarray) longitude
    cLat: latitude
    iy: (int) year in CHAZ iteration
    ichaz: (int) ensemble in CHAZ iteration
    exp: (str) directory name containing historical and future simulations 
    ''' 
    global climInitDate,climInitLon,climInitLat
    climInitDate,climInitLon,climInitLat = cDate,cLon,cLat
    nnt = np.int_(31)                                         # longest track time
    nS = climInitLon.shape[0]

    global fstlon, fstlat, fstldmask
    fstlon = np.zeros([nnt*24+1,nS])*np.nan
    fstlat = np.zeros([nnt*24+1,nS])*np.nan
    fstldmask = np.zeros(fstlat.shape)
    diS = da.from_array(np.arange(0,nS,1).astype(dtype=np.int32),chunks=(1,))
    niS = np.arange(0,nS,1).astype(dtype=np.int32)
    new = da.map_blocks(bam, diS, chunks=(1,), dtype=diS.dtype)
    #with ProgressBar():
    b = new.compute(scheduler='synchronous', num_workers=5)
    

    if not gv.quiet: print('removeland')

    ### is this a bit that needs to be fixed? 
    #sys.exit()
    fstlon = fstlon[::6,:]
    fstlat = fstlat[::6,:]
    fstldmask = fstldmask[::6,:]

    new = da.map_blocks(removeland,diS,chunks=(1,), dtype=diS.dtype)
    c = new.compute(scheduler='synchronous',num_workers=5)
    
    if not gv.quiet: print('give times')
    ### give times
    # TODO - area for speedup
    fsttime = np.empty(fstlon.shape, dtype=object)
    fsttime[0,:] = climInitDate
    dummy = pd.DataFrame(fstlon)

    mask = dummy.notna()
    iT1 = np.int16(mask.idxmax(axis=0)) + 1
    iT2 = np.int16(mask[::-1].idxmax(axis=0))
   
    # Handle all-NaN columns
    iT1[~mask.any(axis=0)] = 0
    iT2[~mask.any(axis=0)] = 0

    ## original 
    for iS in niS:
        fsttime[iT1[iS]:iT2[iS]+1,iS] = \
        [climInitDate[iS]+timedelta(hours=6*iit) for iit in range(iT1[iS],iT2[iS]+1,1)]

        fst = {'lon':fstlon,
               'lat':fstlat,
               'Time':fsttime,
               'ldmask':fstldmask}

    return fst

## removed exp
def calPredictors(fst, iy, ichaz):
    '''
    This function creates the 'trackPredictorsbt' pickle file.
    fst: object
    iy: (int) year in CHAZ.py iteration
    ichaz: (int) ensemble in CHAZ.py iteration
    exp: (str) directory containing historical or future simulations
    '''
    global bt
    time1 = time.time()
    bt = fst2bt(fst)
    nS = fst['lon'].shape[1]
    #diS = da.from_array(np.int32(np.arange(0,nS,1)),chunks=(1,))
    #niS = np.int32(np.arange(0,nS,1))
    diS = da.from_array(np.arange(0,nS,1).astype(dtype=np.int32),chunks=(1,))
    niS = np.arange(0,nS,1).astype(dtype=np.int32)
    new = da.map_blocks(getSpeedDir,diS,chunks=(1,), dtype=diS.dtype)
    #with ProgressBar():
    new.compute(scheduler='synchronous',num_workers=5)
    if not gv.quiet: print('done translation speed'), time.time()-time1
    del new
    #gc.collect()

    time1 = time.time()
    new = da.map_blocks(get_predictors,diS,chunks=(1,), dtype=diS.dtype)
    #with ProgressBar():
    a = new.compute(scheduler='synchronous',num_workers=5)
    if not gv.quiet: print(' done calPredictors'), time.time()-time1
    gc.collect()
    with open(gv.output_path+'trackPredictorsbt'+int2str(iy,4)+'_ens'+int2str(ichaz,3)+'.pik','wb+')as f:
        pickle.dump(bt,f)
    #with open(gv.output_path+'trackPredictorsbt'+int2str(iy,4)+'_ens'+int2str(ichaz,3)+'.pik','w+')as f:
    #    pickle.dump(bt,f)
    #f.close()
    return
