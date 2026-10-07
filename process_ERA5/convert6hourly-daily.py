#!/bin/usr/env python
import numpy as np
import xarray as xr
import sys
import matplotlib.pyplot as plt

import os

### removes _6hrly files after converting to dailymean
DELETE_PROCESSED_FILES = True

def int2str(num,l):
        """
        Given an integer num and desired length l, returns the str(num)
        with appended leading zeroes to match the size l.
        """
        if len(str(num))<l:
                return (l-len(str(num)))*'0'+str(num)
        else:
                return str(num)

def fix_coords(ds):
	"""
    rename 'pressure_level' -> 'level'.
	rename 'valid_time' -> 'time'
    
    Parameters:
	ds (xr.Dataset)
    
    Returns: ds (xr.Dataset)
    """
	## level
	if "pressure_level" in ds.coords or "pressure_level" in ds.dims:
		ds = ds.rename({"pressure_level": "level"})
	elif "level" in ds.coords or "level" in ds.dims:
		pass 
	else:
		raise ValueError(
			"Dataset has neither a 'level' nor 'pressure_level' coordinate."
		)

	## time
	if "valid_time" in ds.coords or "valid_time" in ds.dims:
		ds = ds.rename({"valid_time": "time"})
	elif "time" in ds.coords or "time" in ds.dims:
		pass  # already correctly named
	else:
		raise ValueError(
			"Dataset has neither a 'time' nor 'valid_time' coordinate."
		)	
	return ds

for iy in range(1979,2015):
#for iy in range(2000, 2010):
	for im in range(1,13):
		## TODO: adjust so it does both u and v components? was with only u when i opened
		filename = 'u_component_of_wind_'+int2str(iy,4)+int2str(im,2)+'_6hly.nc'
		try:
			print (filename)
			save_fname = filename[:-7]+'dailymean.nc'
			if not os.path.exists(save_fname):
				ds = fix_coords(xr.open_dataset(filename))
				ds_mean = ds.groupby('time.day').mean('time')
				
				ds_mean.to_netcdf(save_fname)
				#ds_mean.to_netcdf(filename[:-7]+'dailymean.nc')
				if DELETE_PROCESSED_FILES:
					os.remove(filename)
					print(f'{filename} removed')

			else:
				print(save_fname+' already exists, skipped.')   
		except:
			print(f'failed, check {filename}')

		#########
		#v = ds_mean.v.values
		#v1 = ds.v.values
		#v2 = np.nanmean(vl[0:4],axis=0)
		#plt.pcolormesh(v[0,0])
		#plt.contour(v2[0])
		########
	
