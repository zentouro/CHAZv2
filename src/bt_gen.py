
### #!/usr/bin/env python
###########################
## Generate input data   ##
## for CHAZ from IBTrACS ##
###########################


### Necessary Packages
import xarray as xr
import numpy as np
import pandas as pd

from datetime import datetime
from netCDF4 import Dataset

import os

## for IBTraCS file
import Namelist as gv



BT_FIELDS = [
    'PIwspd', 'PIslp', 'PIwspdMean', 'dPIwspd', 'PIslpMean',
    'UShearMean', 'UShear', 'VShearMean', 'VShear',
    'div200Mean', 'div200', 'T200Mean', 'T200',
    'rh500_300', 'rh500_300Mean', 'rhMean', 'rh',
    'T100Mean', 'T100', 'dThetaEMean', 'dThetaE', 'dThetaEsMean', 'dThetaEs',
    'landmask', 'landmaskMean']

class read_ibtracs_v4_xr(object):
    '''
    From IBTrACS use USA agency (e.g., usa_lat, usa_lon, etc.)
    atl - NHC ATL
    enp - NHC ENP
    wnp,sh,ni - JTWC for the rest: SH, IO, and WPC
    'global' reads all data
    No longer need 'gap', but only keep data at 00,06,12,18z...
    '''

    def __init__(self, ncFileName, basins):
        ds = xr.open_dataset(ncFileName)
 
        sourceN = []
        if 'atl' in basins:
            sourceN.extend(['NA'])
        if 'wnp' in basins:
            sourceN.extend(['WP'])
        if 'sh' in basins:
            sourceN.extend(['SA', 'SP', 'SI'])
        if 'ni' in basins:
            sourceN.extend(['NI'])
        if 'enp' in basins:
            sourceN.extend(['EP'])
        if 'global' in basins:
            sourceN = ['NA', 'WP', 'SA', 'SP', 'SI', 'NI', 'EP']
        sourceN = np.array(sourceN)
        #print(sourceN)
 
        basin_col0 = ds.basin[:, 0].values
        arg = []
        for i in range(sourceN.shape[0]):
            arg.extend(np.argwhere(basin_col0 == sourceN[i].encode('UTF-8')).ravel().tolist())
        arg = np.array(arg)
 
        lon = ds.usa_lon.values[arg]
        lon[lon < 0] = lon[lon < 0] + 360
        nNaN = np.argwhere(np.nanmax(np.array(lon), axis=1) != -9999.).ravel()
        ds1 = ds.sel(storm=arg[nNaN])
 
        hours = ds1.time.dt.hour.values
        arg_a, arg_b = np.where((hours == 0) | (hours == 6) | (hours == 12) | (hours == 18))
 
        storm_ids = np.unique(arg_a)
        n_storms = storm_ids.shape[0]
        d1 = int(hours.shape[1] / 2)
 
        lon = np.full([n_storms, d1], np.nan)
        lat = np.full([n_storms, d1], np.nan)
        wspd = np.full([n_storms, d1], np.nan)
        pres = np.full([n_storms, d1], np.nan)
        dist2land = np.full([n_storms, d1], np.nan)
        trspeed = np.full([n_storms, d1], np.nan)
        trdir = np.full([n_storms, d1], np.nan)
        #sdate = np.full([n_storms, d1], np.datetime64('NaT'))
        sdate = np.full([n_storms, d1], np.datetime64('NaT'), dtype=ds1.time.values.dtype)
        status = np.full([n_storms, d1], b'  ', dtype='S2')
 
        for i, iarg in enumerate(storm_ids):
            iarg_b = np.sort(arg_b[arg_a == iarg])
            n = iarg_b.shape[0]
            lon[i, :n] = ds1.usa_lon[iarg, iarg_b].values
            lat[i, :n] = ds1.usa_lat[iarg, iarg_b].values
            wspd[i, :n] = ds1.usa_wind[iarg, iarg_b].values
            pres[i, :n] = ds1.usa_pres[iarg, iarg_b].values
            dist2land[i, :n] = ds1.dist2land[iarg, iarg_b].values
            trspeed[i, :n] = ds1.storm_speed[iarg, iarg_b].values
            trdir[i, :n] = ds1.storm_dir[iarg, iarg_b].values
            sdate[i, :n] = ds1.time[iarg, iarg_b].values
            status[i, :n] = ds1.usa_status[iarg, iarg_b].values
 
        ## convert any leftover IBTrACS fill values to NaN
        for arr in (lon, lat, wspd, pres, dist2land, trspeed, trdir):
            arr[arr == -9999.] = np.nan
 
        self.lon = lon
        self.lat = lat
        self.wspd = wspd
        self.pres = pres
        self.dates = sdate
        self.names = ds1.name[storm_ids].values
        self.stormID = ds1.number[storm_ids].values.astype(float)
        self.distland = dist2land
        self.trspeed = trspeed
        self.trdir = trdir
        self.season = ds1.season[storm_ids].values.astype(float)
        #self.basin = ds1.basin[storm_ids].values
        self.basin = ds1.basin[storm_ids, 0].values
        self.status = status


def _time_to_numeric(dates_2d, clock_start_time=np.datetime64('1800-01-01')):
#def _time_to_numeric(dates_2d, clock_start_time=datetime(1800,1,1,0)):
    """
    Converts a (d1,d2) datetime64 array (NaT = missing) into days since
    `clock_start_time`
 
    TODO: double-check this is the correct start time.
    This is 
    """
    days = (dates_2d - clock_start_time) / np.timedelta64(1, 'D')
    return np.where(pd.isna(dates_2d), 0.0, days).astype(float)

def build_bt(ncFileName, basinName, outFileName):
    '''
    reads in ibtracs information, creates best tracks file for each basin and the globe
    generates empty nan dataarrays for all relevant variables for CHAZ that are later calculated with 
    get_predictors. 
    '''

    r = read_ibtracs_v4_xr(ncFileName, [basinName])

    ## build out bt outputs
    StormLon = r.lon.T
    StormLat = r.lat.T
    StormMwspd = r.wspd.T
    StormMslp = r.pres.T
    trSpeed = r.trspeed.T
    trDir = r.trdir.T
    #dist2land = r.distland.T           ## unused in CHAZ
    dates = r.dates.T  # (d1, d2)
    basin = r.basin

    d1, d2 = StormLon.shape
    if d2 == 0:
        raise ValueError(f"No storms found for basin '{basinName}'")
 
    StormYYYY = np.full((d1, d2), np.nan)
    StormMM = np.full((d1, d2), np.nan)
    StormDD = np.full((d1, d2), np.nan)
    StormHH = np.full((d1, d2), np.nan)
    StormInitMonth = np.full(d2, np.nan)
 
    for j in range(d2):
        col = pd.DatetimeIndex(dates[:, j])
        valid = ~col.isna()
        if valid.any():
            StormYYYY[valid, j] = col.year[valid]
            StormMM[valid, j] = col.month[valid]
            StormDD[valid, j] = col.day[valid]
            StormHH[valid, j] = col.hour[valid]
            StormInitMonth[j] = col.month[valid][0]
 
    StormId = r.stormID
    ## season just outputs the year, confusingly! 
    StormYear = r.season 


    dVdt = np.full((d1, d2), np.nan)
    dPdt = np.full((d1, d2), np.nan)
    dVdt[2:, :] = StormMwspd[2:, :] - StormMwspd[:-2, :]
    dPdt[1:, :] = StormMslp[1:, :] - StormMslp[:-1, :]

    ## conversion, not totally sure about this yet
    Time = _time_to_numeric(dates)

    ## start structure for netcdf output
    data_vars = dict(
        StormId=(("d2",), StormId),
        StormYear=(("d2",), StormYear),
        StormInitMonth=(("d2",), StormInitMonth),
        StormLon=(("d1", "d2"), StormLon),
        StormLat=(("d1", "d2"), StormLat),
        StormMwspd=(("d1", "d2"), StormMwspd),
        StormMslp=(("d1", "d2"), StormMslp),
        Time=(("d1", "d2"), Time),
        trSpeed=(("d1", "d2"), trSpeed),
        trDir=(("d1", "d2"), trDir),
        dVdt=(("d1", "d2"), dVdt),
        dPdt=(("d1", "d2"), dPdt),
        basin=(("d2"), basin)
    )

    ## add empty placeholders for the predictors
    for f in BT_FIELDS:
        data_vars[f] = (("d1", "d2"), np.full((d1, d2), np.nan))

    if 'global' in basinName:
        data_vars['errors'] = (("d1", "d2"), np.full((d1, d2), np.nan))
        
        ## in original, but not used/necessary in CHAZ, neglecting
        #d3 = 10
        #data_vars['predictors'] = (("d1", "d2", "d3"), np.full((d1, d2, d3), np.nan))
        #data_vars['predictand'] = (("d1", "d2"), np.full((d1, d2), np.nan))
        ## also dropped trueY
                                      
    ds_out = xr.Dataset(data_vars=data_vars)
    #assert len(ds_out.data_vars) == 36, f"expected 36 variables, got {len(ds_out.data_vars)}"
    ds_out.to_netcdf(outFileName)
    print(f"wrote {outFileName}  (d1={d1}, d2={d2}, {len(ds_out.data_vars)} variables)")
    return ds_out
 
 
def build_all_basins(ncFileName, basins, outdir):
    for b in basins:
        outFileName = f"{outdir.rstrip('/')}/bt_{b}.nc"
        
        ## If Namelist file sets overwrite to false
        ## AND if bt_*.nc for basin b exists 
        if not gv.overwrite and os.path.exists(outFileName):
            print(f'{outFileName} already exists, skipping')
            pass

        ## If the file doesn't exist or 
        ## overwrite is set to True
        else:
            print(f'starting {b}')
            build_bt(ncFileName, b, outFileName)
 

#if __name__ == "__main__":
#NC_FILE = '/xpt/taroko.local/data0/clee/bt/IBTrACS.ALL.v04r00.nc'
#NC_FILE = "/home/miriamn/CHAZ/CHAZvectorized/input/IBTrACS.ALL.v04r01.nc"
#OUTDIR = "/home/miriamn/CHAZ/CHAZvectorized/input/updating/"

def run_btgen():
    if __name__ == "__main__":
        IBTrACS_FILE = gv.ipath+'IBTrACS.ALL.v04r01.nc'
        OUTDIR = gv.ipath
        build_all_basins(IBTrACS_FILE, ['atl', 'enp', 'wnp', 'sh', 'ni', 'global'], OUTDIR)
    