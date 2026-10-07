
import numpy as np
from datetime import datetime #as real_datetime
import sys
import subprocess
from tools.util import int2str, fix_coords
from scipy.interpolate import interp1d
from calendar import monthrange
from netCDF4 import Dataset
import gc
import Namelist as gv
import pandas as pd
import xarray as xr

import os 
#import time

### ORIGINAL
def createNetCDF(covMatrix,iy,xlong,xlat):
	var = ['u200p2D','v200p2D','u850p2D','v850p2D']
	#nc = Dataset(gv.pre_path+'Cov_'+int2str(iy,4)+'.nc','w',format='NETCDF3_CLASSIC')
	#nc = Dataset(gv.pre_path+'Cov_'+int2str(iy,4)+'.nc','w', format='NETCDF4')
	cov_fname = get_cov_fname(iy)
	nc = Dataset(cov_fname, 'w', format = 'NETCDF4')
	
	nc.createDimension('latitude',xlat.shape[0])
	nc.createDimension('longitude',xlong.shape[0])
	nc.createDimension('month',covMatrix.shape[1])
	lats = nc.createVariable('latitude',np.dtype('float32').char,('latitude',))
	lons = nc.createVariable('longitude',np.dtype('float32').char,('longitude',))
	months = nc.createVariable('month',np.dtype('int32').char,('month',))
	lats.units = 'degrees_north'
	lons.units = 'degrees_east'
	months.units = 'month_in_year'
	lats[:] = xlat
	lons[:] = xlong
	months[:] = range(1,covMatrix.shape[1]+1,1)

	### start to create variables
	count = 0
	for iv in range(len(var)):
		for iiv in range(iv,len(var),1):
			vname = var[iv][:-2]+var[iiv][:-2]
			#print vname
			var1 = nc.createVariable(vname,np.dtype('float32').char,('month','latitude','longitude'))
			var1.units = '(ms-1)^2'
			var1[:] = covMatrix[count,:,:,:]
			count += 1
	nc.close()
	return()

def fillinNaN(var,neighbors):
	"""
	Replace masked areas using interpolation.
	"""
	for ii in range(var.shape[0]):
		a = var[ii,:,:]

		##TODO: how is count being used here? 
		## I think it can be removed
		#count = 0
		while np.any(a.mask):
			a_copy = a.copy()
			for hor_shift,vert_shift in neighbors:
				if not np.any(a.mask): break
				a_shifted=np.roll(a_copy,shift=hor_shift,axis=1)
				a_shifted=np.roll(a_shifted,shift=vert_shift,axis=0)
				idx=~a_shifted.mask*a.mask
				#print count, idx[idx==True].shape 
				a[idx]=a_shifted[idx]
			#count+=1
		var[ii,:,:] = a
	return var

def my_cov(x,y,naxis):
	n = x.shape[naxis]
	cov_bias = np.mean(x*y,axis=naxis)-(np.mean(x,axis=naxis)*np.mean(y,axis=naxis))
	cov_bias = cov_bias*n/(n-1)
	return cov_bias


## could add a pass through tools.util.py/fix_coords
def preprocess(ds):
	'''
	preprocess for xarray open_mfdataset to handle messy ERA5 data
    may not always be needed
	'''
	## drop 'number' coord if present 
	if 'number' in ds.coords:
		ds = ds.drop_vars('number')

	## order level and dtype across all files
	ds = ds.sortby('level')
	ds['level'] = ds['level'].astype('int32')
    
	return ds

def get_cov_fname(iy):
	'''
	Simple function to keep covariance matrix naming the same 
	across two other functions
	'''
	return gv.pre_path + 'Cov_' + int2str(iy, 4) + '.nc'


def run_windCov():
	##### reading monthly data through a list
	##### daily_csv should be in the namelist.py
	# y1 = gv.Year1
	# y2 = gv.Year2
	monthly_csv = gv.monthlycsv
	with open(monthly_csv,'r') as f:
		df_mary = f.readlines()

	df_sub_uam = []
	df_sub_vam = []
	df_sub_hurm = []
	df_sub_uad = []
	df_sub_vad = []
                                                                        
	for ia in df_mary:
		#print(ia)
		if 'u_component' in ia and 'daily' not in ia:
			df_sub_uam.append(ia[:-1])
		if 'u_component' in ia and 'dailymean' in ia:
			df_sub_uad.append(ia[:-1])
		if 'v_component' in ia and 'daily' not in ia:
			df_sub_vam.append(ia[:-1])
		if 'v_component' in ia and 'dailymean' in ia:
			df_sub_vad.append(ia[:-1])
		if 'relative_humidity' in ia:
			df_sub_hurm.append(ia[:-1])
	df_sub_uam = np.array(df_sub_uam)
	df_sub_vam = np.array(df_sub_vam)
	df_sub_hurm = np.array(df_sub_hurm)
	df_sub_uad = np.array(df_sub_uad)
	df_sub_vad = np.array(df_sub_vad)

	## splits string, depends on correct naming 
	modely1m = np.int_([df_sub_uam[i].split('/')[-1].split('_')[-1][:4] for i in range(df_sub_uam.shape[0])])
	modely2m = np.int_([df_sub_uam[i].split('/')[-1].split('_')[-1].split('-')[-1][:4] for i in range(df_sub_uam.shape[0])])

	##### reading daily data through Lamont URL
	##### daily_csv should be in the namelist.py
	modely1d = np.int_([df_sub_vad[i].split('/')[-1].split('_')[-2][:4] for i in range(df_sub_vad.shape[0])])
	modely2d = np.int_([df_sub_vad[i].split('/')[-1].split('_')[-2][:4] for i in range(df_sub_vad.shape[0])])

	arg_y1 = -1
	neighbors=((0,1),(0,-1),(1,0),(-1,0),(1,1),(-1,1),(1,-1),(-1,-1),(0,2),(0,-2),(2,0),(-2,0))
	
	for iy in range(gv.Year1, gv.Year2+1):
		## ovewrite protection:
		## if Covariance Matrix export for a given year already exists
		## and ovewrite = False in Namelist.py, then skip to the next year 
		cov_fname = get_cov_fname(iy)
		if os.path.exists(cov_fname) and not gv.overwrite:
			print(f'{cov_fname} exists, skipping {iy}')
			## then continue on to next year in the for loop
			continue

		## If the file doesn't exist or 
        ## overwrite is set to True

		arg_yd = np.argwhere((modely1d<=iy)&(modely2d>=iy)).ravel()
		arg_ym = np.argwhere((modely1m<=iy)&(modely2m>=iy)).ravel()[0]  

		# arg_ym0 = np.argwhere((modely1m<=iy)&(modely2m>=iy)).ravel().tolist()
		# arg_ym1 = np.argwhere((modely1m<=iy+1)&(modely2m>=iy+1)).ravel().tolist()
		# arg_ym2 = np.argwhere((modely1m<=iy-1)&(modely2m>=iy-1)).ravel().tolist()
		# arg_ym0.extend(arg_ym1)
		# arg_ym0.extend(arg_ym2)
		# arg_ym = np.unique(np.array(arg_ym0))

		if arg_yd[0] != arg_y1:
			## added fix_coords (to handle ERA5 data with new coordinate names)
			ds_uam = fix_coords(xr.open_dataset(df_sub_uam[arg_ym]))
			ds_vam = fix_coords(xr.open_dataset(df_sub_vam[arg_ym]))
			arg_p250m = np.argwhere(ds_uam.level.values==250.).ravel()[0]
			arg_p850m = np.argwhere(ds_uam.level.values==850.).ravel()[0]

			## added preprocess
			ds_uad = xr.open_mfdataset(df_sub_uad[arg_yd], preprocess = preprocess, combine='nested', concat_dim='day')
			ds_vad = xr.open_mfdataset(df_sub_vad[arg_yd], preprocess = preprocess, combine='nested', concat_dim='day')

			## debugging
			# print(ds_vad.dims)
			# print(ds_vad.day.shape)
			# print(len(df_sub_vad[arg_yd]))   # how many daily files matched this year

			arg_p250d = np.argwhere(ds_vad.level.values==250.).ravel()[0]
			arg_p850d = np.argwhere(ds_vad.level.values==850.).ravel()[0]
			xlong = ds_vam.longitude.values
			xlat = ds_vam.latitude.values
			dsvadtime = pd.to_datetime(np.arange(ds_vad.day.shape[0]),unit='D',origin=pd.Timestamp(str(iy)+'-01-01'))  

		###### This is a better way but xr.interP does not support chunk in the interpolation axis
		#arg_tm = np.argwhere((ds_uam.time.dt.year.values==iy)|(ds_uam.time.dt.year.values==iy-1)|(ds_uam.time.dt.year.values==iy+1)).ravel()
		arg_tm = np.argwhere((ds_uam.time.dt.year.values==iy)).ravel()
		arg_td = np.argwhere((dsvadtime.year.values==iy)).ravel()
		### cftime.DatetimeNoLeap  #### THIS IS REALLY for CESM2's calendar?? 
		missing_value = 1e+20

		date_daily = dsvadtime[arg_td]
		date_monthly = ds_uam.time[arg_tm]

		### from /data0/clee/ERA5/pre
		ndays_year = 0
		m = []
		for im in range(1,13):
			m.append([monthrange(iy,im)[1]/2+ndays_year])
			ndays_year = ndays_year+monthrange(iy,im)[1]
		m = np.squeeze(np.int_(m)-1)

		### on github
		#m = np.argwhere(date_daily.day==15).ravel() ## roughly middle of the month

		ua250m = ds_uam.u[arg_tm,arg_p250m].values
		ua850m = ds_uam.u[arg_tm,arg_p850m].values
		ua850m[ua850m!=ua850m] = 1e+20
		ua850m = np.ma.masked_values(ua850m, missing_value)
		ua850m = fillinNaN(ua850m,neighbors)
		va250m = ds_vam.v[arg_tm,arg_p250m].values
		va850m = ds_vam.v[arg_tm,arg_p850m].values
		va850m[va850m!=va850m] = 1e+20
		va850m = np.ma.masked_values(va850m, missing_value)
		va850m = fillinNaN(va850m,neighbors)
		d = np.arange(0,date_daily.shape[0])
		f = interp1d(m,ua250m,bounds_error=False,fill_value="extrapolate",axis=0)
		ua250md = f(d)
		f = interp1d(m,va250m,bounds_error=False,fill_value="extrapolate",axis=0)
		va250md = f(d)
		f = interp1d(m,ua850m,bounds_error=False,fill_value="extrapolate",axis=0)
		ua850md = f(d)
		f = interp1d(m,va850m,bounds_error=False,fill_value="extrapolate",axis=0)
		va850md = f(d)
		
		##### now we are going to do the monthly cov.
		covMatrix = np.zeros([10,12,va250md.shape[1],va250md.shape[2]])
		for im in range(1,13):
			arg_td = np.argwhere((dsvadtime.year==iy)&(dsvadtime.month==im)).ravel()
			ua250d = ds_uad.u[arg_td,arg_p250d].values
			ua850d = ds_uad.u[arg_td,arg_p850d].values
			va250d = ds_vad.v[arg_td,arg_p250d].values
			va850d = ds_vad.v[arg_td,arg_p850d].values
			#missing_values = nc.variables['ua'].missing_value
			missing_value = 1e+20
			ua850d[ua850d!=ua850d] = 1e+20
			va850d[va850d!=va850d] = 1e+20
			va850d = np.ma.masked_values(va850d, missing_value)
			ua850d = np.ma.masked_values(ua850d, missing_value)
			ua850d = fillinNaN(ua850d,neighbors)
			va850d = fillinNaN(va850d,neighbors)

		
			arg_md = np.argwhere((dsvadtime[arg_td].year==iy)&(dsvadtime[arg_td].month==im)).ravel()
			### debugging
			# print(f'arg_td: {arg_td}')
			# print(f'arg_md: {arg_md}')

			u250p = ua250d-ua250md[arg_md]
			u250p2D = u250p.reshape([u250p.shape[0],u250p.shape[1]*u250p.shape[2]])
			u850p = ua850d-ua850md[arg_md]
			u850p2D = u850p.reshape([u250p.shape[0],u250p.shape[1]*u250p.shape[2]])
			v250p = va250d-va250md[arg_md]
			v250p2D = v250p.reshape([u250p.shape[0],u250p.shape[1]*u250p.shape[2]])
			v850p = va850d-va850md[arg_md]
			v850p2D = v850p.reshape([u250p.shape[0],u250p.shape[1]*u250p.shape[2]])
			var = ['u250p2D','v250p2D','u850p2D','v850p2D']
			count = 0
			for iv in range(len(var)):
				vname = var[iv]
				for iiv in range(iv,len(var),1):
					vname1 = var[iiv]
					covMatrix[count,im-1,:,:] = \
						my_cov(eval(vname),eval(vname1),0).reshape([u250p.shape[1],u250p.shape[2]])
						#np.hstack([np.cov(eval(vname)[:,igrid],eval(vname1)[:,igrid])[0,1] \
						#for igrid in range(u250p2D.shape[1])]).reshape([u250p.shape[1],u250p.shape[2]])
					count += 1
					
			### only printout if quiet not suppressed
			if gv.quiet:
				print(iy, im, count, vname, vname1)

		createNetCDF(covMatrix,iy,xlong,xlat)
		del covMatrix,u250p, u250p2D, u850p, u850p2D, v250p, v250p2D, v850p, v850p2D  
		gc.collect()



### TODO: REMOVE COMMENTED COPIES IF UNECESSARY 
### UPDATE
# def createNetCDF(covMatrix, iy, xlong, xlat):
#     """Write covariance matrix. Uses a single nc.variables dict write per var."""
#     var = ['u200p2D','v200p2D','u850p2D','v850p2D']
#     #nc = Dataset(gv.pre_path+'Cov_'+int2str(iy,4)+'.nc', 'w', format='NETCDF3_CLASSIC')
#     nc = Dataset(gv.pre_path+'Cov_'+int2str(iy,4)+'.nc', 'w')
#     nc.createDimension('latitude', xlat.shape[0])
#     nc.createDimension('longitude', xlong.shape[0])
#     nc.createDimension('month', covMatrix.shape[1])
#     lats = nc.createVariable('latitude', np.dtype('float32').char, ('latitude',))
#     lons = nc.createVariable('longitude', np.dtype('float32').char, ('longitude',))
#     months = nc.createVariable('month', np.dtype('int32').char, ('month',))
#     lats.units = 'degrees_north'
#     lons.units = 'degrees_east'
#     months.units = 'month_in_year'
#     lats[:] = xlat
#     lons[:] = xlong
#     months[:] = range(1, covMatrix.shape[1]+1, 1)

#     count = 0
#     for iv in range(len(var)):
#         for iiv in range(iv, len(var), 1):
#             vname = var[iv][:-2]+var[iiv][:-2]
#             var1 = nc.createVariable(vname, np.dtype('float32').char,
#                                      ('month','latitude','longitude'),
#                                      # FIX 1: chunking + compression cuts write time significantly
#                                      zlib=False,  # NETCDF3_CLASSIC doesn't support zlib
#                                      )
#             var1.units = '(ms-1)^2'
#             var1[:] = covMatrix[count,:,:,:]
#             count += 1
#     nc.close()
#     return()

# def fillinNaN(var,neighbors):
#     for ii in range(var.shape[0]):
#         a = var[ii,:,:]
#         count = 0
#         while np.any(a.mask):
#             a_copy = a.copy()
#             for hor_shift,vert_shift in neighbors:
#                 if not np.any(a.mask): break
#                 a_shifted=np.roll(a_copy,shift=hor_shift,axis=1)
#                 a_shifted=np.roll(a_shifted,shift=vert_shift,axis=0)
#                 idx=~a_shifted.mask*a.mask
#                 a[idx]=a_shifted[idx]
#             count+=1
#         var[ii,:,:] = a
#     return var

# def my_cov(x,y,naxis):
#     n = x.shape[naxis]
#     cov_bias = np.mean(x*y,axis=naxis)-(np.mean(x,axis=naxis)*np.mean(y,axis=naxis))
#     cov_bias = cov_bias*n/(n-1)
#     return cov_bias

# def my_cov_all(fields):
#     """
#     FIX 2: Compute all 10 pairwise covariances in one pass.
#     fields: list of 4 arrays, each shape (ntime, nlat*nlon)
#     Returns array of shape (10, nlat*nlon)
#     """
#     n = fields[0].shape[0]
#     count = 0
#     results = []
#     for iv in range(len(fields)):
#         for iiv in range(iv, len(fields)):
#             x = fields[iv]
#             y = fields[iiv]
#             cov = (np.mean(x*y, axis=0) - np.mean(x, axis=0)*np.mean(y, axis=0)) * n/(n-1)
#             results.append(cov)
#             count += 1
#     return np.stack(results, axis=0)  # (10, nlat*nlon)

# def run_windCov():
#     t_total = time.time()

#     y1 = gv.Year1
#     y2 = gv.Year2
#     monthly_csv = gv.monthlycsv
#     with open(monthly_csv,'r') as f:
#         df_mary = f.readlines()

#     df_sub_uam = []
#     df_sub_vam = []
#     df_sub_hurm = []
#     df_sub_uad = []
#     df_sub_vad = []
                                                                    
#     for ia in df_mary:
#         if 'u_component' in ia and 'daily' not in ia:
#             df_sub_uam.append(ia[:-1])
#         if 'u_component' in ia and 'dailymean' in ia:
#             df_sub_uad.append(ia[:-1])
#         if 'v_component' in ia and 'daily' not in ia:
#             df_sub_vam.append(ia[:-1])
#         if 'v_component' in ia and 'dailymean' in ia:
#             df_sub_vad.append(ia[:-1])
#         if 'relative_humidity' in ia:
#             df_sub_hurm.append(ia[:-1])
#     df_sub_uam = np.array(df_sub_uam)
#     df_sub_vam = np.array(df_sub_vam)
#     df_sub_hurm = np.array(df_sub_hurm)
#     df_sub_uad = np.array(df_sub_uad)
#     df_sub_vad = np.array(df_sub_vad)
#     modely1m = np.int_([df_sub_uam[i].split('/')[-1].split('_')[-1][:4] for i in range(df_sub_uam.shape[0])])
#     modely2m = np.int_([df_sub_uam[i].split('/')[-1].split('_')[-1].split('-')[-1][:4] for i in range(df_sub_uam.shape[0])])

#     modely1d = np.int_([df_sub_vad[i].split('/')[-1].split('_')[-2][:4] for i in range(df_sub_vad.shape[0])])
#     modely2d = np.int_([df_sub_vad[i].split('/')[-1].split('_')[-2][:4] for i in range(df_sub_vad.shape[0])])

#     arg_y1 = -1
#     neighbors=((0,1),(0,-1),(1,0),(-1,0),(1,1),(-1,1),(1,-1),(-1,-1),(0,2),(0,-2),(2,0),(-2,0))
#     for iy in range(gv.Year1,gv.Year2+1):
#         print(f"\n=== Year {iy} ===")
#         t_year = time.time()

#         arg_yd = np.argwhere((modely1d<=iy)&(modely2d>=iy)).ravel()
#         arg_ym = np.argwhere((modely1m<=iy)&(modely2m>=iy)).ravel()[0]

#         t0 = time.time()
#         if arg_yd[0] != arg_y1:
#             ds_uam = xr.open_dataset(df_sub_uam[arg_ym])
#             ds_vam = xr.open_dataset(df_sub_vam[arg_ym])
#             arg_p250m = np.argwhere(ds_uam.level.values==250.).ravel()[0]
#             arg_p850m = np.argwhere(ds_uam.level.values==850.).ravel()[0]
#             ds_uad = xr.open_mfdataset(df_sub_uad[arg_yd])
#             ds_vad = xr.open_mfdataset(df_sub_vad[arg_yd])
#             arg_p250d = np.argwhere(ds_vad.level.values==250.).ravel()[0]
#             arg_p850d = np.argwhere(ds_vad.level.values==850.).ravel()[0]
#             xlong = ds_vam.longitude.values
#             xlat = ds_vam.latitude.values
#             dsvadtime = pd.to_datetime(np.arange(ds_vad.day.shape[0]),unit='D',origin=pd.Timestamp(str(iy)+'-01-01'))
#         print(f"  [open datasets]        {time.time()-t0:.2f}s")

#         arg_tm = np.argwhere((ds_uam.time.dt.year.values==iy)).ravel()
#         arg_td = np.argwhere((dsvadtime.year.values==iy)).ravel()
#         missing_value = 1e+20

#         date_daily = dsvadtime[arg_td]
#         date_monthly = ds_uam.time[arg_tm]

#         ndays_year = 0
#         m = []
#         for im in range(1,13):
#                 m.append([monthrange(iy,im)[1]/2+ndays_year])
#                 ndays_year = ndays_year+monthrange(iy,im)[1]
#         m = np.squeeze(np.int_(m)-1)

#         t0 = time.time()
#         ua250m = ds_uam.u[arg_tm,arg_p250m].values
#         ua850m = ds_uam.u[arg_tm,arg_p850m].values
#         ua850m[ua850m!=ua850m] = 1e+20
#         ua850m = np.ma.masked_values(ua850m, missing_value)
#         ua850m = fillinNaN(ua850m,neighbors)
#         va250m = ds_vam.v[arg_tm,arg_p250m].values
#         va850m = ds_vam.v[arg_tm,arg_p850m].values
#         va850m[va850m!=va850m] = 1e+20
#         va850m = np.ma.masked_values(va850m, missing_value)
#         va850m = fillinNaN(va850m,neighbors)
#         print(f"  [load+fill monthly]    {time.time()-t0:.2f}s")

#         # FIX 3: reshape to 2D before interp1d so it works on (ntime, ngrid)
#         # instead of (ntime, nlat, nlon) — same result, avoids internal reshape overhead
#         nlat, nlon = ua250m.shape[1], ua250m.shape[2]
#         ngrid = nlat * nlon
#         ua250m_2d = np.array(ua250m).reshape(12, ngrid)
#         va250m_2d = np.array(va250m).reshape(12, ngrid)
#         ua850m_2d = np.array(ua850m).reshape(12, ngrid)
#         va850m_2d = np.array(va850m).reshape(12, ngrid)

#         t0 = time.time()
#         d = np.arange(0, date_daily.shape[0])
#         f = interp1d(m, ua250m_2d, bounds_error=False, fill_value="extrapolate", axis=0)
#         ua250md = f(d)   # (ndays, ngrid)
#         f = interp1d(m, va250m_2d, bounds_error=False, fill_value="extrapolate", axis=0)
#         va250md = f(d)
#         f = interp1d(m, ua850m_2d, bounds_error=False, fill_value="extrapolate", axis=0)
#         ua850md = f(d)
#         f = interp1d(m, va850m_2d, bounds_error=False, fill_value="extrapolate", axis=0)
#         va850md = f(d)
#         print(f"  [interpolate monthly]  {time.time()-t0:.2f}s")

#         covMatrix = np.zeros([10,12,nlat,nlon])
#         for im in range(1,13):
#             t0 = time.time()
#             arg_td = np.argwhere((dsvadtime.year==iy)&(dsvadtime.month==im)).ravel()
#             ua250d = ds_uad.u[arg_td,arg_p250d].values
#             ua850d = ds_uad.u[arg_td,arg_p850d].values
#             va250d = ds_vad.v[arg_td,arg_p250d].values
#             va850d = ds_vad.v[arg_td,arg_p850d].values
#             t_load = time.time()-t0

#             t0 = time.time()
#             missing_value = 1e+20
#             ua850d[ua850d!=ua850d] = 1e+20
#             va850d[va850d!=va850d] = 1e+20
#             va850d = np.ma.masked_values(va850d, missing_value)
#             ua850d = np.ma.masked_values(ua850d, missing_value)
#             ua850d = fillinNaN(ua850d,neighbors)
#             va850d = fillinNaN(va850d,neighbors)
#             t_fill = time.time()-t0

#             t0 = time.time()
#             arg_md = np.argwhere((dsvadtime[arg_td].year==iy)&(dsvadtime[arg_td].month==im)).ravel()

#             # FIX 2: replace eval() loop with my_cov_all on pre-flattened 2D arrays
#             u250p2D = ua250d.reshape(ua250d.shape[0], ngrid) - ua250md[arg_md]
#             v250p2D = va250d.reshape(va250d.shape[0], ngrid) - va250md[arg_md]
#             u850p2D = np.array(ua850d).reshape(ua850d.shape[0], ngrid) - ua850md[arg_md]
#             v850p2D = np.array(va850d).reshape(va850d.shape[0], ngrid) - va850md[arg_md]

#             cov_flat = my_cov_all([u250p2D, v250p2D, u850p2D, v850p2D])  # (10, ngrid)
#             covMatrix[:, im-1, :, :] = cov_flat.reshape(10, nlat, nlon)
#             t_cov = time.time()-t0

#             print(f"  month {im:02d}: load={t_load:.2f}s  fill={t_fill:.2f}s  cov={t_cov:.2f}s")

#         t0 = time.time()
#         createNetCDF(covMatrix,iy,xlong,xlat)
#         print(f"  [write NetCDF]         {time.time()-t0:.2f}s")

#         print(f"  [year total]           {time.time()-t_year:.2f}s")
#         del covMatrix, u250p2D, u850p2D, v250p2D, v850p2D
#         gc.collect()

#     print(f"\n[grand total] {time.time()-t_total:.2f}s")







##### TRIAL AND ERROR 
# import numpy as np
# from datetime import datetime as real_datetime
# import sys
# import subprocess
# from tools.util import int2str
# from scipy.interpolate import interp1d
# from calendar import monthrange
# from netCDF4 import Dataset
# import gc
# import Namelist as gv
# import pandas as pd
# import xarray as xr
# import time

# def createNetCDF(covMatrix,iy,xlong,xlat):
# 	var = ['u200p2D','v200p2D','u850p2D','v850p2D']
# 	nc = Dataset(gv.pre_path+'Cov_'+int2str(iy,4)+'.nc','w',format='NETCDF3_CLASSIC')
# 	nc.createDimension('latitude',xlat.shape[0])
# 	nc.createDimension('longitude',xlong.shape[0])
# 	nc.createDimension('month',covMatrix.shape[1])
# 	lats = nc.createVariable('latitude',np.dtype('float32').char,('latitude',))
# 	lons = nc.createVariable('longitude',np.dtype('float32').char,('longitude',))
# 	months = nc.createVariable('month',np.dtype('int32').char,('month',))
# 	lats.units = 'degrees_north'
# 	lons.units = 'degrees_east'
# 	months.units = 'month_in_year'
# 	lats[:] = xlat
# 	lons[:] = xlong
# 	months[:] = range(1,covMatrix.shape[1]+1,1)

# 	### start to create variables
# 	count = 0
# 	for iv in range(len(var)):
# 		for iiv in range(iv,len(var),1):
# 			vname = var[iv][:-2]+var[iiv][:-2]
# 			var1 = nc.createVariable(vname,np.dtype('float32').char,('month','latitude','longitude'))
# 			var1.units = '(ms-1)^2'
# 			var1[:] = covMatrix[count,:,:,:]
# 			count += 1
# 	nc.close()
# 	return()

# def fillinNaN(var,neighbors):
# 	for ii in range(var.shape[0]):
# 		a = var[ii,:,:]
# 		count = 0
# 		while np.any(a.mask):
# 			a_copy = a.copy()
# 			for hor_shift,vert_shift in neighbors:
# 				if not np.any(a.mask): break
# 				a_shifted=np.roll(a_copy,shift=hor_shift,axis=1)
# 				a_shifted=np.roll(a_shifted,shift=vert_shift,axis=0)
# 				idx=~a_shifted.mask*a.mask
# 				a[idx]=a_shifted[idx]
# 			count+=1
# 		var[ii,:,:] = a
# 	return var

# def my_cov(x,y,naxis):
# 	n = x.shape[naxis]
# 	cov_bias = np.mean(x*y,axis=naxis)-(np.mean(x,axis=naxis)*np.mean(y,axis=naxis))
# 	cov_bias = cov_bias*n/(n-1)
# 	return cov_bias

# def run_windCov():
# 	t_total = time.time()

# 	y1 = gv.Year1
# 	y2 = gv.Year2
# 	monthly_csv = gv.monthlycsv
# 	with open(monthly_csv,'r') as f:
# 		df_mary = f.readlines()

# 	df_sub_uam = []
# 	df_sub_vam = []
# 	df_sub_hurm = []
# 	df_sub_uad = []
# 	df_sub_vad = []
                                                                        
# 	for ia in df_mary:
# 		if 'u_component' in ia and 'daily' not in ia:
# 			df_sub_uam.append(ia[:-1])
# 		if 'u_component' in ia and 'dailymean' in ia:
# 			df_sub_uad.append(ia[:-1])
# 		if 'v_component' in ia and 'daily' not in ia:
# 			df_sub_vam.append(ia[:-1])
# 		if 'v_component' in ia and 'dailymean' in ia:
# 			df_sub_vad.append(ia[:-1])
# 		if 'relative_humidity' in ia:
# 			df_sub_hurm.append(ia[:-1])
# 	df_sub_uam = np.array(df_sub_uam)
# 	df_sub_vam = np.array(df_sub_vam)
# 	df_sub_hurm = np.array(df_sub_hurm)
# 	df_sub_uad = np.array(df_sub_uad)
# 	df_sub_vad = np.array(df_sub_vad)
# 	modely1m = np.int_([df_sub_uam[i].split('/')[-1].split('_')[-1][:4] for i in range(df_sub_uam.shape[0])])
# 	modely2m = np.int_([df_sub_uam[i].split('/')[-1].split('_')[-1].split('-')[-1][:4] for i in range(df_sub_uam.shape[0])])

# 	modely1d = np.int_([df_sub_vad[i].split('/')[-1].split('_')[-2][:4] for i in range(df_sub_vad.shape[0])])
# 	modely2d = np.int_([df_sub_vad[i].split('/')[-1].split('_')[-2][:4] for i in range(df_sub_vad.shape[0])])

# 	arg_y1 = -1
# 	neighbors=((0,1),(0,-1),(1,0),(-1,0),(1,1),(-1,1),(1,-1),(-1,-1),(0,2),(0,-2),(2,0),(-2,0))
# 	for iy in range(gv.Year1,gv.Year2+1):
# 		print(f"\n=== Year {iy} ===")
# 		t_year = time.time()

# 		arg_yd = np.argwhere((modely1d<=iy)&(modely2d>=iy)).ravel()
# 		arg_ym = np.argwhere((modely1m<=iy)&(modely2m>=iy)).ravel()[0]

# 		t0 = time.time()
# 		if arg_yd[0] != arg_y1:
# 			ds_uam = xr.open_dataset(df_sub_uam[arg_ym])
# 			ds_vam = xr.open_dataset(df_sub_vam[arg_ym])
# 			arg_p250m = np.argwhere(ds_uam.level.values==250.).ravel()[0]
# 			arg_p850m = np.argwhere(ds_uam.level.values==850.).ravel()[0]
# 			ds_uad = xr.open_mfdataset(df_sub_uad[arg_yd])
# 			ds_vad = xr.open_mfdataset(df_sub_vad[arg_yd])
# 			arg_p250d = np.argwhere(ds_vad.level.values==250.).ravel()[0]
# 			arg_p850d = np.argwhere(ds_vad.level.values==850.).ravel()[0]
# 			xlong = ds_vam.longitude.values
# 			xlat = ds_vam.latitude.values
# 			dsvadtime = pd.to_datetime(np.arange(ds_vad.day.shape[0]),unit='D',origin=pd.Timestamp(str(iy)+'-01-01'))
# 		print(f"  [open datasets]        {time.time()-t0:.2f}s")

# 		arg_tm = np.argwhere((ds_uam.time.dt.year.values==iy)).ravel()
# 		arg_td = np.argwhere((dsvadtime.year.values==iy)).ravel()
# 		missing_value = 1e+20

# 		date_daily = dsvadtime[arg_td]
# 		date_monthly = ds_uam.time[arg_tm]

# 		ndays_year = 0
# 		m = []
# 		for im in range(1,13):
# 				m.append([monthrange(iy,im)[1]/2+ndays_year])
# 				ndays_year = ndays_year+monthrange(iy,im)[1]
# 		m = np.squeeze(np.int_(m)-1)

# 		t0 = time.time()
# 		ua250m = ds_uam.u[arg_tm,arg_p250m].values
# 		ua850m = ds_uam.u[arg_tm,arg_p850m].values
# 		ua850m[ua850m!=ua850m] = 1e+20
# 		ua850m = np.ma.masked_values(ua850m, missing_value)
# 		ua850m = fillinNaN(ua850m,neighbors)
# 		va250m = ds_vam.v[arg_tm,arg_p250m].values
# 		va850m = ds_vam.v[arg_tm,arg_p850m].values
# 		va850m[va850m!=va850m] = 1e+20
# 		va850m = np.ma.masked_values(va850m, missing_value)
# 		va850m = fillinNaN(va850m,neighbors)
# 		print(f"  [load+fill monthly]    {time.time()-t0:.2f}s")

# 		t0 = time.time()
# 		d = np.arange(0,date_daily.shape[0])
# 		f = interp1d(m,ua250m,bounds_error=False,fill_value="extrapolate",axis=0)
# 		ua250md = f(d)
# 		f = interp1d(m,va250m,bounds_error=False,fill_value="extrapolate",axis=0)
# 		va250md = f(d)
# 		f = interp1d(m,ua850m,bounds_error=False,fill_value="extrapolate",axis=0)
# 		ua850md = f(d)
# 		f = interp1d(m,va850m,bounds_error=False,fill_value="extrapolate",axis=0)
# 		va850md = f(d)
# 		print(f"  [interpolate monthly]  {time.time()-t0:.2f}s")

# 		covMatrix = np.zeros([10,12,va250md.shape[1],va250md.shape[2]])
# 		for im in range(1,13):
# 			t0 = time.time()
# 			arg_td = np.argwhere((dsvadtime.year==iy)&(dsvadtime.month==im)).ravel()
# 			ua250d = ds_uad.u[arg_td,arg_p250d].values
# 			ua850d = ds_uad.u[arg_td,arg_p850d].values
# 			va250d = ds_vad.v[arg_td,arg_p250d].values
# 			va850d = ds_vad.v[arg_td,arg_p850d].values
# 			t_load = time.time()-t0

# 			t0 = time.time()
# 			missing_value = 1e+20
# 			ua850d[ua850d!=ua850d] = 1e+20
# 			va850d[va850d!=va850d] = 1e+20
# 			va850d = np.ma.masked_values(va850d, missing_value)
# 			ua850d = np.ma.masked_values(ua850d, missing_value)
# 			ua850d = fillinNaN(ua850d,neighbors)
# 			va850d = fillinNaN(va850d,neighbors)
# 			t_fill = time.time()-t0

# 			t0 = time.time()
# 			arg_md = np.argwhere((dsvadtime[arg_td].year==iy)&(dsvadtime[arg_td].month==im)).ravel()
# 			u250p = ua250d-ua250md[arg_md]
# 			u250p2D = u250p.reshape([u250p.shape[0],u250p.shape[1]*u250p.shape[2]])
# 			u850p = ua850d-ua850md[arg_md]
# 			u850p2D = u850p.reshape([u250p.shape[0],u250p.shape[1]*u250p.shape[2]])
# 			v250p = va250d-va250md[arg_md]
# 			v250p2D = v250p.reshape([u250p.shape[0],u250p.shape[1]*u250p.shape[2]])
# 			v850p = va850d-va850md[arg_md]
# 			v850p2D = v850p.reshape([u250p.shape[0],u250p.shape[1]*u250p.shape[2]])
# 			var = ['u250p2D','v250p2D','u850p2D','v850p2D']
# 			count = 0
# 			for iv in range(len(var)):
# 				vname = var[iv]
# 				for iiv in range(iv,len(var),1):
# 					vname1 = var[iiv]
# 					covMatrix[count,im-1,:,:] = \
# 						my_cov(eval(vname),eval(vname1),0).reshape([u250p.shape[1],u250p.shape[2]])
# 					count += 1
# 			t_cov = time.time()-t0

# 			print(f"  month {im:02d}: load={t_load:.2f}s  fill={t_fill:.2f}s  cov={t_cov:.2f}s")

# 		t0 = time.time()
# 		createNetCDF(covMatrix,iy,xlong,xlat)
# 		print(f"  [write NetCDF]         {time.time()-t0:.2f}s")

# 		print(f"  [year total]           {time.time()-t_year:.2f}s")
# 		del covMatrix,u250p, u250p2D, u850p, u850p2D, v250p, v250p2D, v850p, v850p2D  
# 		gc.collect()

# 	print(f"\n[grand total] {time.time()-t_total:.2f}s")

# # import gc
# # import numpy as np
# # import pandas as pd
# # import xarray as xr
# # from calendar import monthrange
# # from netCDF4 import Dataset
# # from scipy.interpolate import interp1d

# # import Namelist as gv
# # from tools.util import int2str


# # # ---------------------------------------------------------------------------
# # # Constants
# # # ---------------------------------------------------------------------------
# # MISSING = 1e20
# # NEIGHBORS = (
# #     (0, 1), (0, -1), (1, 0), (-1, 0),
# #     (1, 1), (-1, 1), (1, -1), (-1, -1),
# #     (0, 2), (0, -2), (2, 0), (-2, 0),
# # )
# # PRESSURE_LEVELS = (250., 850.)
# # WIND_VARS = ["u250p2D", "v250p2D", "u850p2D", "v850p2D"]


# # # ---------------------------------------------------------------------------
# # # NetCDF output
# # # ---------------------------------------------------------------------------
# # def createNetCDF(covMatrix, iy, xlong, xlat):
# #     """Write the 10-element covariance matrix to a NetCDF3 file."""
# #     var_labels = ["u200p2D", "v200p2D", "u850p2D", "v850p2D"]  # kept as-is from original
# #     path = gv.pre_path + "Cov_" + int2str(iy, 4) + ".nc"

# #     with Dataset(path, "w", format="NETCDF3_CLASSIC") as nc:
# #         nc.createDimension("latitude",  xlat.shape[0])
# #         nc.createDimension("longitude", xlong.shape[0])
# #         nc.createDimension("month",     covMatrix.shape[1])

# #         lats   = nc.createVariable("latitude",  "f4", ("latitude",))
# #         lons   = nc.createVariable("longitude", "f4", ("longitude",))
# #         months = nc.createVariable("month",     "i4", ("month",))

# #         lats.units   = "degrees_north"
# #         lons.units   = "degrees_east"
# #         months.units = "month_in_year"

# #         lats[:]   = xlat
# #         lons[:]   = xlong
# #         months[:] = np.arange(1, covMatrix.shape[1] + 1)

# #         count = 0
# #         for iv in range(len(var_labels)):
# #             for iiv in range(iv, len(var_labels)):
# #                 vname = var_labels[iv][:-2] + var_labels[iiv][:-2]
# #                 v = nc.createVariable(vname, "f4", ("month", "latitude", "longitude"))
# #                 v.units = "(ms-1)^2"
# #                 v[:] = covMatrix[count]
# #                 count += 1


# # # ---------------------------------------------------------------------------
# # # Gap-filling
# # # ---------------------------------------------------------------------------
# # def fillinNaN(var, neighbors):
# #     """
# #     Fill masked grid cells by spreading values from valid neighbours.
# #     Operates in-place and also returns the array.
# #     """
# #     for ii in range(var.shape[0]):
# #         a = var[ii]
# #         while np.any(a.mask):
# #             a_copy = a.copy()
# #             for hor_shift, vert_shift in neighbors:
# #                 if not np.any(a.mask):
# #                     break
# #                 a_shifted = np.roll(np.roll(a_copy, hor_shift, axis=1), vert_shift, axis=0)
# #                 fill_idx = ~a_shifted.mask & a.mask
# #                 a[fill_idx] = a_shifted[fill_idx]
# #         var[ii] = a
# #     return var


# # # ---------------------------------------------------------------------------
# # # Covariance helpers
# # # ---------------------------------------------------------------------------
# # def unbiased_cov(x, y, axis):
# #     """Unbiased sample covariance along `axis` (matches original my_cov)."""
# #     n = x.shape[axis]
# #     return (np.mean(x * y, axis=axis) - np.mean(x, axis=axis) * np.mean(y, axis=axis)) * n / (n - 1)


# # def compute_cov_matrix(u250, v250, u850, v850):
# #     """
# #     Compute all 10 pairwise covariances at once using einsum.

# #     Replaces the nested iv/iiv loop with eval() inside the month loop.
# #     Returns array of shape (10, nlat, nlon).

# #     Pairs (matching original order):
# #       0  u250-u250
# #       1  u250-v250
# #       2  u250-u850
# #       3  u250-v850
# #       4  v250-v250
# #       5  v250-u850
# #       6  v250-v850
# #       7  u850-u850
# #       8  u850-v850
# #       9  v850-v850
# #     """
# #     fields = [u250, v250, u850, v850]  # each shape: (ndays, nlat, nlon)
# #     n = fields[0].shape[0]
# #     pairs = [(i, j) for i in range(4) for j in range(i, 4)]  # 10 pairs

# #     # Stack to (4, ndays, nlat, nlon) for vectorised ops
# #     F = np.stack(fields, axis=0)                    # (4, T, H, W)
# #     F_mean = F.mean(axis=1, keepdims=True)          # (4, 1, H, W)
# #     F_anom = F - F_mean                             # (4, T, H, W)

# #     cov_all = np.stack(
# #         [
# #             np.einsum("thw,thw->hw", F_anom[i], F_anom[j]) / (n - 1)
# #             for i, j in pairs
# #         ],
# #         axis=0,
# #     )  # (10, H, W)
# #     return cov_all


# # # ---------------------------------------------------------------------------
# # # Masking / filling shortcut
# # # ---------------------------------------------------------------------------
# # def mask_and_fill(arr, neighbors, missing=MISSING):
# #     """Replace NaN with missing, mask, and neighbour-fill."""
# #     arr[np.isnan(arr)] = missing
# #     arr = np.ma.masked_values(arr, missing)
# #     return fillinNaN(arr, neighbors)


# # # ---------------------------------------------------------------------------
# # # Mid-month day indices for interpolation
# # # ---------------------------------------------------------------------------
# # def midmonth_day_indices(year):
# #     """
# #     Return an integer array of length 12 giving the (0-based) day index
# #     of the mid-point of each calendar month for `year`.
# #     """
# #     indices, running = [], 0
# #     for month in range(1, 13):
# #         ndays = monthrange(year, month)[1]
# #         indices.append(int(ndays / 2) + running)
# #         running += ndays
# #     return np.array(indices) - 1   # -1 for 0-based indexing (matches original)


# # # ---------------------------------------------------------------------------
# # # Main routine
# # # ---------------------------------------------------------------------------
# # def run_windCov():
# #     # ------------------------------------------------------------------
# #     # 1. Parse the monthly-data CSV list
# #     # ------------------------------------------------------------------
# #     with open(gv.monthlycsv, "r") as f:
# #         lines = [l.rstrip() for l in f]

# #     def grep(lst, *include, exclude=None):
# #         result = [l for l in lst if all(k in l for k in include)]
# #         if exclude:
# #             result = [l for l in result if exclude not in l]
# #         return np.array(result)

# #     uam_files  = grep(lines, "u_component", exclude="daily")
# #     vam_files  = grep(lines, "v_component", exclude="daily")
# #     uad_files  = grep(lines, "u_component", "dailymean")
# #     vad_files  = grep(lines, "v_component", "dailymean")

# #     # Year ranges encoded in filenames
# #     def file_year(files, part, pos):
# #         return np.array([int(f.split("/")[-1].split("_")[part][:4].split("-")[pos]) for f in files])

# #     uam_y1 = file_year(uam_files, -1,  0)
# #     uam_y2 = file_year(uam_files, -1, -1)   # works because split('-')[-1][:4]
# #     vad_y1 = file_year(vad_files, -2,  0)
# #     vad_y2 = file_year(vad_files, -2,  0)   # original used same position for both

# #     # ------------------------------------------------------------------
# #     # 2. Year loop
# #     # ------------------------------------------------------------------
# #     prev_daily_arg = -1

# #     for iy in range(gv.Year1, gv.Year2 + 1):

# #         # Locate file indices for this year
# #         daily_idx   = np.flatnonzero((vad_y1 <= iy) & (vad_y2 >= iy))
# #         monthly_idx = np.flatnonzero((uam_y1 <= iy) & (uam_y2 >= iy))[0]

# #         # Only re-open datasets when the daily file set changes
# #         if daily_idx[0] != prev_daily_arg:
# #             ds_uam = xr.open_dataset(uam_files[monthly_idx])
# #             ds_vam = xr.open_dataset(vam_files[monthly_idx])

# #             p250m = int(np.flatnonzero(ds_uam.level.values == 250.)[0])
# #             p850m = int(np.flatnonzero(ds_uam.level.values == 850.)[0])

# #             ds_uad = xr.open_mfdataset(uad_files[daily_idx])
# #             ds_vad = xr.open_mfdataset(vad_files[daily_idx])

# #             p250d = int(np.flatnonzero(ds_vad.level.values == 250.)[0])
# #             p850d = int(np.flatnonzero(ds_vad.level.values == 850.)[0])

# #             xlong = ds_vam.longitude.values
# #             xlat  = ds_vam.latitude.values

# #             daily_dates = pd.to_datetime(
# #                 np.arange(ds_vad.day.shape[0]), unit="D",
# #                 origin=pd.Timestamp(f"{iy}-01-01"),
# #             )
# #             prev_daily_arg = daily_idx[0]

# #         # Time-slice indices for this year
# #         tm_idx = np.flatnonzero(ds_uam.time.dt.year.values == iy)
# #         td_idx = np.flatnonzero(daily_dates.year == iy)

# #         # ------------------------------------------------------------------
# #         # 3. Interpolate monthly means onto daily grid
# #         # ------------------------------------------------------------------
# #         mid_days = midmonth_day_indices(iy)
# #         d_axis   = np.arange(len(td_idx))

# #         # Load monthly wind fields once
# #         ua250m = ds_uam.u[tm_idx, p250m].values
# #         va250m = ds_vam.v[tm_idx, p250m].values
# #         ua850m = mask_and_fill(ds_uam.u[tm_idx, p850m].values, NEIGHBORS)
# #         va850m = mask_and_fill(ds_vam.v[tm_idx, p850m].values, NEIGHBORS)

# #         def interp_to_daily(monthly_field):
# #             f = interp1d(mid_days, monthly_field, axis=0,
# #                          bounds_error=False, fill_value="extrapolate")
# #             return f(d_axis)

# #         ua250_daily = interp_to_daily(ua250m)
# #         va250_daily = interp_to_daily(va250m)
# #         ua850_daily = interp_to_daily(ua850m)
# #         va850_daily = interp_to_daily(va850m)

# #         # ------------------------------------------------------------------
# #         # 4. Monthly covariance loop
# #         # ------------------------------------------------------------------
# #         nlat, nlon = ua250_daily.shape[1], ua250_daily.shape[2]
# #         covMatrix  = np.zeros((10, 12, nlat, nlon))

# #         for im in range(1, 13):
# #             month_mask = (daily_dates.year == iy) & (daily_dates.month == im)
# #             md_idx = np.flatnonzero(month_mask)   # indices into full daily array
# #             local_idx = np.flatnonzero(           # indices into the year-slice
# #                 (daily_dates[td_idx].year == iy) & (daily_dates[td_idx].month == im)
# #             )

# #             # Daily anomalies = raw daily - interpolated monthly mean
# #             u250_anom = ds_uad.u[md_idx, p250d].values - ua250_daily[local_idx]
# #             v250_anom = ds_vad.v[md_idx, p250d].values - va250_daily[local_idx]
# #             u850_anom = mask_and_fill(
# #                 ds_uad.u[md_idx, p850d].values - ua850_daily[local_idx], NEIGHBORS
# #             )
# #             v850_anom = mask_and_fill(
# #                 ds_vad.v[md_idx, p850d].values - va850_daily[local_idx], NEIGHBORS
# #             )

# #             covMatrix[:, im - 1] = compute_cov_matrix(
# #                 u250_anom, v250_anom, u850_anom, v850_anom
# #             )
# #             print(iy, im)

# #         createNetCDF(covMatrix, iy, xlong, xlat)

# #         # Explicit cleanup keeps memory stable across years
# #         del covMatrix, ua250_daily, va250_daily, ua850_daily, va850_daily
# #         del u250_anom, v250_anom, u850_anom, v850_anom
# #         gc.collect()



# # # #!/usr/bin/env python
# # # import numpy as np
# # # from datetime import datetime as real_datetime
# # # #from netcdftime import utime,datetime		## not used as far as i can tell
# # # import sys
# # # import subprocess
# # # from tools.util import int2str
# # # from scipy.interpolate import interp1d
# # # from calendar import monthrange
# # # from netCDF4 import Dataset
# # # import gc
# # # import Namelist as gv
# # # import pandas as pd
# # # import xarray as xr

# # # def createNetCDF(covMatrix,iy,xlong,xlat):
# # # 	var = ['u200p2D','v200p2D','u850p2D','v850p2D']
# # # 	nc = Dataset(gv.pre_path+'Cov_'+int2str(iy,4)+'.nc','w',format='NETCDF3_CLASSIC')
# # # 	nc.createDimension('latitude',xlat.shape[0])
# # # 	nc.createDimension('longitude',xlong.shape[0])
# # # 	nc.createDimension('month',covMatrix.shape[1])
# # # 	lats = nc.createVariable('latitude',np.dtype('float32').char,('latitude',))
# # # 	lons = nc.createVariable('longitude',np.dtype('float32').char,('longitude',))
# # # 	months = nc.createVariable('month',np.dtype('int32').char,('month',))
# # # 	lats.units = 'degrees_north'
# # # 	lons.units = 'degrees_east'
# # # 	months.units = 'month_in_year'
# # # 	lats[:] = xlat
# # # 	lons[:] = xlong
# # # 	months[:] = range(1,covMatrix.shape[1]+1,1)

# # # 	### start to create variables
# # # 	count = 0
# # # 	for iv in range(len(var)):
# # # 		for iiv in range(iv,len(var),1):
# # # 			vname = var[iv][:-2]+var[iiv][:-2]
# # # 			#print vname
# # # 			var1 = nc.createVariable(vname,np.dtype('float32').char,('month','latitude','longitude'))
# # # 			var1.units = '(ms-1)^2'
# # # 			var1[:] = covMatrix[count,:,:,:]
# # # 			count += 1
# # # 	nc.close()
# # # 	return()

# # # def fillinNaN(var,neighbors):
# # # 	"""
# # # 	Replace masked areas using interpolation.
# # # 	"""
# # # 	for ii in range(var.shape[0]):
# # # 		a = var[ii,:,:]

# # # 		##TODO: how is count being used here? 
# # # 		## I think it can be removed
# # # 		count = 0
# # # 		while np.any(a.mask):
# # # 			a_copy = a.copy()
# # # 			for hor_shift,vert_shift in neighbors:
# # # 				if not np.any(a.mask): break
# # # 				a_shifted=np.roll(a_copy,shift=hor_shift,axis=1)
# # # 				a_shifted=np.roll(a_shifted,shift=vert_shift,axis=0)
# # # 				idx=~a_shifted.mask*a.mask
# # # 				#print count, idx[idx==True].shape 
# # # 				a[idx]=a_shifted[idx]
# # # 			count+=1
# # # 		var[ii,:,:] = a
# # # 	return var

# # # def my_cov(x,y,naxis):
# # # 	n = x.shape[naxis]
# # # 	cov_bias = np.mean(x*y,axis=naxis)-(np.mean(x,axis=naxis)*np.mean(y,axis=naxis))
# # # 	cov_bias = cov_bias*n/(n-1)
# # # 	return cov_bias

# # # def run_windCov():
# # # 	##### reading monthly data through a list
# # # 	##### daily_csv should be in the namelist.py
# # # 	y1 = gv.Year1
# # # 	y2 = gv.Year2
# # # 	monthly_csv = gv.monthlycsv
# # # 	with open(monthly_csv,'r') as f:
# # # 		df_mary = f.readlines()

# # # 	df_sub_uam = []
# # # 	df_sub_vam = []
# # # 	df_sub_hurm = []
# # # 	df_sub_uad = []
# # # 	df_sub_vad = []
                                                                        
# # # 	for ia in df_mary:
# # # 		#print(ia)
# # # 		if 'u_component' in ia and 'daily' not in ia:
# # # 			df_sub_uam.append(ia[:-1])
# # # 		if 'u_component' in ia and 'dailymean' in ia:
# # # 			df_sub_uad.append(ia[:-1])
# # # 		if 'v_component' in ia and 'daily' not in ia:
# # # 			df_sub_vam.append(ia[:-1])
# # # 		if 'v_component' in ia and 'dailymean' in ia:
# # # 			df_sub_vad.append(ia[:-1])
# # # 		if 'relative_humidity' in ia:
# # # 			df_sub_hurm.append(ia[:-1])
# # # 	df_sub_uam = np.array(df_sub_uam)
# # # 	df_sub_vam = np.array(df_sub_vam)
# # # 	df_sub_hurm = np.array(df_sub_hurm)
# # # 	df_sub_uad = np.array(df_sub_uad)
# # # 	df_sub_vad = np.array(df_sub_vad)
# # # 	modely1m = np.int_([df_sub_uam[i].split('/')[-1].split('_')[-1][:4] for i in range(df_sub_uam.shape[0])])
# # # 	modely2m = np.int_([df_sub_uam[i].split('/')[-1].split('_')[-1].split('-')[-1][:4] for i in range(df_sub_uam.shape[0])])

# # # 	##### reading daily data through Lamont URL
# # # 	##### daily_csv should be in the namelist.py
# # # 	modely1d = np.int_([df_sub_vad[i].split('/')[-1].split('_')[-2][:4] for i in range(df_sub_vad.shape[0])])
# # # 	modely2d = np.int_([df_sub_vad[i].split('/')[-1].split('_')[-2][:4] for i in range(df_sub_vad.shape[0])])

# # # 	arg_y1 = -1
# # # 	neighbors=((0,1),(0,-1),(1,0),(-1,0),(1,1),(-1,1),(1,-1),(-1,-1),(0,2),(0,-2),(2,0),(-2,0))
# # # 	for iy in range(gv.Year1,gv.Year2+1):
# # # 		arg_yd = np.argwhere((modely1d<=iy)&(modely2d>=iy)).ravel()
# # # 		arg_ym = np.argwhere((modely1m<=iy)&(modely2m>=iy)).ravel()[0]
# # # 		#arg_ym0 = np.argwhere((modely1m<=iy)&(modely2m>=iy)).ravel().tolist()
# # # 		#arg_ym1 = np.argwhere((modely1m<=iy+1)&(modely2m>=iy+1)).ravel().tolist()
# # # 		#arg_ym2 = np.argwhere((modely1m<=iy-1)&(modely2m>=iy-1)).ravel().tolist()
# # # 		#arg_ym0.extend(arg_ym1)
# # # 		#arg_ym0.extend(arg_ym2)
# # # 		#arg_ym = np.unique(np.array(arg_ym0))
# # # 		if arg_yd[0] != arg_y1:
# # # 			ds_uam = xr.open_dataset(df_sub_uam[arg_ym])
# # # 			ds_vam = xr.open_dataset(df_sub_vam[arg_ym])
# # # 			arg_p250m = np.argwhere(ds_uam.level.values==250.).ravel()[0]
# # # 			arg_p850m = np.argwhere(ds_uam.level.values==850.).ravel()[0]
# # # 			ds_uad = xr.open_mfdataset(df_sub_uad[arg_yd])
# # # 			ds_vad = xr.open_mfdataset(df_sub_vad[arg_yd])
# # # 			arg_p250d = np.argwhere(ds_vad.level.values==250.).ravel()[0]
# # # 			arg_p850d = np.argwhere(ds_vad.level.values==850.).ravel()[0]
# # # 			xlong = ds_vam.longitude.values
# # # 			xlat = ds_vam.latitude.values
# # # 			dsvadtime = pd.to_datetime(np.arange(ds_vad.day.shape[0]),unit='D',origin=pd.Timestamp(str(iy)+'-01-01'))
# # # 		###### This is a better way but xr.interP does not support chunk in the interpolation axis
# # # 		#arg_tm = np.argwhere((ds_uam.time.dt.year.values==iy)|(ds_uam.time.dt.year.values==iy-1)|(ds_uam.time.dt.year.values==iy+1)).ravel()
# # # 		arg_tm = np.argwhere((ds_uam.time.dt.year.values==iy)).ravel()
# # # 		arg_td = np.argwhere((dsvadtime.year.values==iy)).ravel()
# # # 		### cftime.DatetimeNoLeap  #### THIS IS REALLY for CESM2's calendar?? 
# # # 		missing_value = 1e+20

# # # 		date_daily = dsvadtime[arg_td]
# # # 		date_monthly =ds_uam.time[arg_tm]

# # # 		### from /data0/clee/ERA5/pre
# # # 		ndays_year = 0
# # # 		m = []
# # # 		for im in range(1,13):
# # # 				m.append([monthrange(iy,im)[1]/2+ndays_year])
# # # 				ndays_year = ndays_year+monthrange(iy,im)[1]
# # # 		m = np.squeeze(np.int_(m)-1)

# # # 		### on github
# # # 		#m = np.argwhere(date_daily.day==15).ravel() ## roughly middle of the month

# # # 		ua250m = ds_uam.u[arg_tm,arg_p250m].values
# # # 		ua850m = ds_uam.u[arg_tm,arg_p850m].values
# # # 		ua850m[ua850m!=ua850m] = 1e+20
# # # 		ua850m = np.ma.masked_values(ua850m, missing_value)
# # # 		ua850m = fillinNaN(ua850m,neighbors)
# # # 		va250m = ds_vam.v[arg_tm,arg_p250m].values
# # # 		va850m = ds_vam.v[arg_tm,arg_p850m].values
# # # 		va850m[va850m!=va850m] = 1e+20
# # # 		va850m = np.ma.masked_values(va850m, missing_value)
# # # 		va850m = fillinNaN(va850m,neighbors)
# # # 		d = np.arange(0,date_daily.shape[0])
# # # 		f = interp1d(m,ua250m,bounds_error=False,fill_value="extrapolate",axis=0)
# # # 		ua250md = f(d)
# # # 		f = interp1d(m,va250m,bounds_error=False,fill_value="extrapolate",axis=0)
# # # 		va250md = f(d)
# # # 		f = interp1d(m,ua850m,bounds_error=False,fill_value="extrapolate",axis=0)
# # # 		ua850md = f(d)
# # # 		f = interp1d(m,va850m,bounds_error=False,fill_value="extrapolate",axis=0)
# # # 		va850md = f(d)
# # # 		##### now we are going to do the monthly cov.
# # # 		covMatrix = np.zeros([10,12,va250md.shape[1],va250md.shape[2]])
# # # 		for im in range(1,13):
# # # 			arg_td = np.argwhere((dsvadtime.year==iy)&(dsvadtime.month==im)).ravel()
# # # 			ua250d = ds_uad.u[arg_td,arg_p250d].values
# # # 			ua850d = ds_uad.u[arg_td,arg_p850d].values
# # # 			va250d = ds_vad.v[arg_td,arg_p250d].values
# # # 			va850d = ds_vad.v[arg_td,arg_p850d].values
# # # 			#missing_values = nc.variables['ua'].missing_value
# # # 			missing_value = 1e+20
# # # 			ua850d[ua850d!=ua850d] = 1e+20
# # # 			va850d[va850d!=va850d] = 1e+20
# # # 			va850d = np.ma.masked_values(va850d, missing_value)
# # # 			ua850d = np.ma.masked_values(ua850d, missing_value)
# # # 			ua850d = fillinNaN(ua850d,neighbors)
# # # 			va850d = fillinNaN(va850d,neighbors)

# # # 			arg_md = np.argwhere((dsvadtime[arg_td].year==iy)&(dsvadtime[arg_td].month==im)).ravel()
# # # 			u250p = ua250d-ua250md[arg_md]
# # # 			u250p2D = u250p.reshape([u250p.shape[0],u250p.shape[1]*u250p.shape[2]])
# # # 			u850p = ua850d-ua850md[arg_md]
# # # 			u850p2D = u850p.reshape([u250p.shape[0],u250p.shape[1]*u250p.shape[2]])
# # # 			v250p = va250d-va250md[arg_md]
# # # 			v250p2D = v250p.reshape([u250p.shape[0],u250p.shape[1]*u250p.shape[2]])
# # # 			v850p = va850d-va850md[arg_md]
# # # 			v850p2D = v850p.reshape([u250p.shape[0],u250p.shape[1]*u250p.shape[2]])
# # # 			var = ['u250p2D','v250p2D','u850p2D','v850p2D']
# # # 			count = 0
# # # 			for iv in range(len(var)):
# # # 				vname = var[iv]
# # # 				for iiv in range(iv,len(var),1):
# # # 					vname1 = var[iiv]
# # # 					covMatrix[count,im-1,:,:] = \
# # # 						my_cov(eval(vname),eval(vname1),0).reshape([u250p.shape[1],u250p.shape[2]])
# # # 						#np.hstack([np.cov(eval(vname)[:,igrid],eval(vname1)[:,igrid])[0,1] \
# # # 						#for igrid in range(u250p2D.shape[1])]).reshape([u250p.shape[1],u250p.shape[2]])
# # # 					count += 1
# # # 			print(iy, im, count, vname, vname1)

# # # 		createNetCDF(covMatrix,iy,xlong,xlat)
# # # 		del covMatrix,u250p, u250p2D, u850p, u850p2D, v250p, v250p2D, v850p, v850p2D  
# # # 		gc.collect()


# # # # #!/usr/bin/env python
# # # # import numpy as np
# # # # from datetime import datetime as real_datetime
# # # # #from netcdftime import utime,datetime		## not used as far as i can tell
# # # # import sys
# # # # import subprocess
# # # # from tools.util import int2str
# # # # from scipy.interpolate import interp1d
# # # # from calendar import monthrange
# # # # from netCDF4 import Dataset
# # # # import gc
# # # # import Namelist as gv
# # # # import pandas as pd
# # # # import xarray as xr

# # # # import time

# # # # def createNetCDF(covMatrix,iy,xlong,xlat):
# # # # 	var = ['u200p2D','v200p2D','u850p2D','v850p2D']
# # # # 	nc = Dataset(gv.pre_path+'Cov_'+int2str(iy,4)+'.nc','w',format='NETCDF3_CLASSIC')
# # # # 	nc.createDimension('latitude',xlat.shape[0])
# # # # 	nc.createDimension('longitude',xlong.shape[0])
# # # # 	nc.createDimension('month',covMatrix.shape[1])
# # # # 	lats = nc.createVariable('latitude',np.dtype('float32').char,('latitude',))
# # # # 	lons = nc.createVariable('longitude',np.dtype('float32').char,('longitude',))
# # # # 	months = nc.createVariable('month',np.dtype('int32').char,('month',))
# # # # 	lats.units = 'degrees_north'
# # # # 	lons.units = 'degrees_east'
# # # # 	months.units = 'month_in_year'
# # # # 	lats[:] = xlat
# # # # 	lons[:] = xlong
# # # # 	months[:] = range(1,covMatrix.shape[1]+1,1)

# # # # 	### start to create variables
# # # # 	count = 0
# # # # 	for iv in range(len(var)):
# # # # 		for iiv in range(iv,len(var),1):
# # # # 			vname = var[iv][:-2]+var[iiv][:-2]
# # # # 			#print vname
# # # # 			var1 = nc.createVariable(vname,np.dtype('float32').char,('month','latitude','longitude'))
# # # # 			var1.units = '(ms-1)^2'
# # # # 			var1[:] = covMatrix[count,:,:,:]
# # # # 			count += 1
# # # # 	nc.close()
# # # # 	return()

# # # # def fillinNaN(var,neighbors):
# # # # 	"""
# # # # 	Replace masked areas using interpolation.
# # # # 	"""
# # # # 	for ii in range(var.shape[0]):
# # # # 		a = var[ii,:,:]

# # # # 		##TODO: how is count being used here? 
# # # # 		## I think it can be removed
# # # # 		count = 0
# # # # 		while np.any(a.mask):
# # # # 			a_copy = a.copy()
# # # # 			for hor_shift,vert_shift in neighbors:
# # # # 				if not np.any(a.mask): break
# # # # 				a_shifted=np.roll(a_copy,shift=hor_shift,axis=1)
# # # # 				a_shifted=np.roll(a_shifted,shift=vert_shift,axis=0)
# # # # 				idx=~a_shifted.mask*a.mask
# # # # 				#print count, idx[idx==True].shape 
# # # # 				a[idx]=a_shifted[idx]
# # # # 			count+=1
# # # # 		var[ii,:,:] = a
# # # # 	return var

# # # # ## new
# # # # def fillinNaN_xr(da: xr.DataArray) -> xr.DataArray:
# # # #     """
# # # #     Replace NaN cells by repeatedly propagating valid neighbors,
# # # #     replicating the roll-based flood fill in fillinNaN().
# # # #     Operates on a DataArray with dims (..., lat, lon) or (time, lat, lon).
# # # #     """
# # # #     # One pass of forward+backward fill along each spatial axis covers
# # # #     # the (0,±1), (±1,0) cardinal neighbors. Repeating converges like
# # # #     # the original while loop, but xarray's ffill/bfill is vectorized
# # # #     # across all time steps simultaneously — no Python loop over ii.
# # # #     max_iter = 20  # safeguard; should converge in <5 for typical land masks
# # # #     for _ in range(max_iter):
# # # #         if not da.isnull().any():
# # # #             break
# # # #         da = da.ffill('lon').bfill('lon').ffill('lat').bfill('lat')
# # # #     return da

# # # # def _mask_and_fill(arr: np.ndarray, da_template: xr.DataArray,
# # # #                    missing_value=1e20) -> np.ndarray:
# # # #     """Wrap a raw numpy array in a DataArray, fill NaN/missing, return numpy."""
# # # #     arr = arr.astype(float)
# # # #     arr[arr == missing_value] = np.nan
# # # #     arr[arr != arr] = np.nan          # catches any remaining IEEE NaN
# # # #     da = xr.DataArray(arr, dims=da_template.dims[:arr.ndim],
# # # #                       coords={k: da_template.coords[k]
# # # #                                for k in da_template.dims[:arr.ndim]
# # # #                                if k in da_template.coords})
# # # #     return fillinNaN_xr(da).values

# # # # ### original
# # # # def my_cov(x,y,naxis):
# # # # 	n = x.shape[naxis]
# # # # 	cov_bias = np.mean(x*y,axis=naxis)-(np.mean(x,axis=naxis)*np.mean(y,axis=naxis))
# # # # 	cov_bias = cov_bias*n/(n-1)
# # # # 	return cov_bias

# # # # # ### updated for vectorizing 
# # # # # def my_cov_stack(stack, naxis=1):
# # # # #     """
# # # # #     Vectorized pairwise covariance matrix for all variable pairs.
# # # # #     """
# # # # #     n     = stack.shape[naxis]
# # # # #     e_xy  = np.einsum('itn,jtn->ijn', stack, stack) / n
# # # # #     e_x   = stack.mean(axis=naxis)        # [nvars, ngrid]
# # # # #     cov_bias_full = (e_xy - np.einsum('in,jn->ijn', e_x, e_x)) * n / (n - 1)
# # # # #     return cov_bias_full


# # # # ## updated
# # # # def run_windCov():
# # # # 	t0 = time.time()

# # # # 	## UNCHANGED
# # # # 	#y1 = gv.Year1
# # # # 	#y2 = gv.Year2
# # # # 	monthly_csv = gv.monthlycsv
# # # # 	with open(monthly_csv, 'r') as f:
# # # # 		df_mary = f.readlines()

# # # # 	df_sub_uam, df_sub_vam, df_sub_hurm = [], [], []
# # # # 	df_sub_uad, df_sub_vad = [], []

# # # # 	for ia in df_mary:
# # # # 		if 'u_component' in ia and 'daily' not in ia:    df_sub_uam.append(ia[:-1])
# # # # 		if 'u_component' in ia and 'dailymean' in ia:    df_sub_uad.append(ia[:-1])
# # # # 		if 'v_component' in ia and 'daily' not in ia:    df_sub_vam.append(ia[:-1])
# # # # 		if 'v_component' in ia and 'dailymean' in ia:    df_sub_vad.append(ia[:-1])
# # # # 		if 'relative_humidity' in ia:                    df_sub_hurm.append(ia[:-1])

# # # # 	df_sub_uam = np.array(df_sub_uam)
# # # # 	df_sub_vam = np.array(df_sub_vam)
# # # # 	df_sub_hurm = np.array(df_sub_hurm)
# # # # 	df_sub_uad = np.array(df_sub_uad)
# # # # 	df_sub_vad = np.array(df_sub_vad)

# # # # 	modely1m = np.int_([df_sub_uam[i].split('/')[-1].split('_')[-1][:4]            for i in range(df_sub_uam.shape[0])])
# # # # 	modely2m = np.int_([df_sub_uam[i].split('/')[-1].split('_')[-1].split('-')[-1][:4] for i in range(df_sub_uam.shape[0])])
# # # # 	modely1d = np.int_([df_sub_vad[i].split('/')[-1].split('_')[-2][:4]            for i in range(df_sub_vad.shape[0])])
# # # # 	modely2d = np.int_([df_sub_vad[i].split('/')[-1].split('_')[-2][:4]            for i in range(df_sub_vad.shape[0])])

# # # # 	t1 = time.time()

# # # # 	arg_y1 = -1
# # # # 	neighbors = ((0,1),(0,-1),(1,0),(-1,0),(1,1),(-1,1),(1,-1),(-1,-1),(0,2),(0,-2),(2,0),(-2,0))
# # # # 	missing_value = 1e+20

# # # # 	# Precompute the upper-triangle index pairs once.
# # # # 	# These replace the nested `for iv / for iiv` loop with eval().
# # # # 	var_names = ['u250p', 'v250p', 'u850p', 'v850p']   # 4 variables → 10 pairs
# # # # 	var_pairs = [(i, j) for i in range(4) for j in range(i, 4)]  # 10 upper-tri pairs

# # # # 	for iy in range(gv.Year1, gv.Year2 + 1):
# # # # 		arg_yd = np.argwhere((modely1d <= iy) & (modely2d >= iy)).ravel()
# # # # 		arg_ym = np.argwhere((modely1m <= iy) & (modely2m >= iy)).ravel()[0]

# # # # 		if arg_yd[0] != arg_y1:
# # # # 			arg_y1 = arg_yd[0]

# # # # 			# --- [UNCHANGED] open monthly datasets ---
# # # # 			ds_uam = xr.open_dataset(df_sub_uam[arg_ym])
# # # # 			ds_vam = xr.open_dataset(df_sub_vam[arg_ym])
# # # # 			arg_p250m = np.argwhere(ds_uam.level.values == 250.).ravel()[0]
# # # # 			arg_p850m = np.argwhere(ds_uam.level.values == 850.).ravel()[0]

# # # # 			t = time.time()
# # # # 			# KEY CHANGE 1: open daily datasets once per file set, not once per month.
# # # # 			# open_mfdataset already concatenates all matching files along 'day'.
# # # # 			ds_uad = xr.open_mfdataset(df_sub_uad[arg_yd])
# # # # 			ds_vad = xr.open_mfdataset(df_sub_vad[arg_yd])
# # # # 			print(f'  open_mfdataset={time.time()-t:.2f}')

# # # # 			arg_p250d = np.argwhere(ds_vad.level.values == 250.).ravel()[0]
# # # # 			arg_p850d = np.argwhere(ds_vad.level.values == 850.).ravel()[0]
# # # # 			xlong = ds_vam.longitude.values
# # # # 			xlat  = ds_vam.latitude.values
# # # # 			dsvadtime = pd.to_datetime(
# # # # 				np.arange(ds_vad.day.shape[0]), unit='D',
# # # # 				origin=pd.Timestamp(str(iy) + '-01-01')
# # # # 			)

# # # # 		arg_tm = np.argwhere(ds_uam.time.dt.year.values == iy).ravel()
# # # # 		arg_td = np.argwhere(dsvadtime.year.values == iy).ravel()

# # # # 		# Build mid-month indices (same logic as before)
# # # # 		ndays_year = 0
# # # # 		m = []
# # # # 		for im in range(1, 13):
# # # # 			m.append([monthrange(iy, im)[1] / 2 + ndays_year])
# # # # 			ndays_year += monthrange(iy, im)[1]
# # # # 		m = np.squeeze(np.int_(m) - 1)

# # # # 		# --- [UNCHANGED] monthly interpolation to daily ---
# # # # 		ua250m = ds_uam.u[arg_tm, arg_p250m].values
# # # # 		ua850m = ds_uam.u[arg_tm, arg_p850m].values
# # # # 		ua850m[ua850m != ua850m] = missing_value
# # # # 		ua850m = fillinNaN(np.ma.masked_values(ua850m, missing_value), neighbors)

# # # # 		va250m = ds_vam.v[arg_tm, arg_p250m].values
# # # # 		va850m = ds_vam.v[arg_tm, arg_p850m].values
# # # # 		va850m[va850m != va850m] = missing_value
# # # # 		va850m = fillinNaN(np.ma.masked_values(va850m, missing_value), neighbors)

# # # # 		d = np.arange(0, dsvadtime[arg_td].shape[0])
# # # # 		ua250md = interp1d(m, ua250m, bounds_error=False, fill_value="extrapolate", axis=0)(d)
# # # # 		va250md = interp1d(m, va250m, bounds_error=False, fill_value="extrapolate", axis=0)(d)
# # # # 		ua850md = interp1d(m, ua850m, bounds_error=False, fill_value="extrapolate", axis=0)(d)
# # # # 		va850md = interp1d(m, va850m, bounds_error=False, fill_value="extrapolate", axis=0)(d)

# # # # 		t = time.time()

# # # # 		ua850d_yr = _mask_and_fill(ds_uad.u[arg_td, arg_p850d].values, ds_uad.u)
# # # # 		va850d_yr = _mask_and_fill(ds_vad.v[arg_td, arg_p850d].values, ds_vad.v)
# # # # 		# (ua250d_yr and va250d_yr have no NaN issues at 250 hPa so skip filling there)

# # # # 		# # KEY CHANGE 2: load ALL daily data for the year in one shot (all months).
# # # # 		# # Previously this happened inside the im loop → 12 separate .values calls.
# # # # 		# ua250d_yr = ds_uad.u[arg_td, arg_p250d].values   # shape: (ndays, nlat, nlon)
# # # # 		# ua850d_yr = ds_uad.u[arg_td, arg_p850d].values
# # # # 		# va250d_yr = ds_vad.v[arg_td, arg_p250d].values
# # # # 		# va850d_yr = ds_vad.v[arg_td, arg_p850d].values

# # # # 		# # Mask/fill NaN on 850 hPa (same logic, applied to full year at once)
# # # # 		# for arr in [ua850d_yr, va850d_yr]:
# # # # 		#     arr[arr != arr] = missing_value
# # # # 		# ua850d_yr = fillinNaN(np.ma.masked_values(ua850d_yr, missing_value), neighbors)
# # # # 		# va850d_yr = fillinNaN(np.ma.masked_values(va850d_yr, missing_value), neighbors)

# # # # 		# KEY CHANGE 3: compute anomalies for the full year at once (no im loop needed).
# # # # 		u250p_yr = ua250d_yr - ua250md   # shape: (ndays, nlat, nlon)
# # # # 		v250p_yr = va250d_yr - va250md
# # # # 		u850p_yr = ua850d_yr - ua850md
# # # # 		v850p_yr = va850d_yr - va850md

# # # # 		nlat, nlon = u250p_yr.shape[1], u250p_yr.shape[2]

# # # # 		# KEY CHANGE 4: stack the 4 anomaly fields into one array → (4, ndays, nlat, nlon).
# # # # 		# This replaces eval() with a clean array index and eliminates the string dispatch.
# # # # 		anom_yr = np.stack([u250p_yr, v250p_yr, u850p_yr, v850p_yr], axis=0)
# # # # 		# Reshape spatial dims for the covariance calculation → (4, ndays, ngrid)
# # # # 		ngrid = nlat * nlon
# # # # 		anom_yr2D = anom_yr.reshape(4, anom_yr.shape[1], ngrid)  # (4, ndays, ngrid)

# # # # 		covMatrix = np.zeros([10, 12, nlat, nlon])

# # # # 		# KEY CHANGE 5: the im loop now only does grouping + the cov call.
# # # # 		# All data loading and anomaly computation moved outside.
# # # # 		for im in range(1, 13):
# # # # 			arg_m = np.argwhere(
# # # # 				(dsvadtime[arg_td].year == iy) & (dsvadtime[arg_td].month == im)
# # # # 			).ravel()

# # # # 			# Slice the pre-loaded anomaly stack to this month's days
# # # # 			anom_m = anom_yr2D[:, arg_m, :]   # shape: (4, ndays_m, ngrid)

# # # # 			# KEY CHANGE 6: compute all 10 upper-triangle covariances at once
# # # # 			# using einsum to form the outer product over the time dimension.
# # # # 			# my_cov(X, Y, axis=0) = mean(X*Y, axis=0) for zero-mean anomalies,
# # # # 			# so we can vectorize: cov[i,j] = mean over time of anom[i] * anom[j].
# # # # 			ndays_m = anom_m.shape[1]
# # # # 			for count, (iv, iiv) in enumerate(var_pairs):
# # # # 				# shape: (ngrid,) → reshaped to (nlat, nlon)
# # # # 				covMatrix[count, im-1, :, :] = (
# # # # 					my_cov(anom_m[iv], anom_m[iiv], 0)
# # # # 					.reshape(nlat, nlon)
# # # # 				)

# # # # 			print(iy, im, len(var_pairs),
# # # # 					var_names[var_pairs[-1][0]], var_names[var_pairs[-1][1]])

# # # # 		print(f'  monthly covariance={time.time()-t:.2f}')

# # # # 		t2 = time.time()
# # # # 		createNetCDF(covMatrix, iy, xlong, xlat)

# # # # 		del covMatrix, anom_yr, anom_yr2D, anom_m
# # # # 		del u250p_yr, v250p_yr, u850p_yr, v850p_yr
# # # # 		del ua250d_yr, va250d_yr, ua850d_yr, va850d_yr

# # # # 		t3 = time.time()
# # # # 		gc.collect()
# # # # 		print(f'calWind run times: import={t1-t0:.2f}  big for loop={t2-t1:.2f}  createNetCDF={t3-t2:.2f}')

# # # # ### original
# # # # def run_windCov_old():
# # # # 	t0 = time.time()

# # # # 	##### reading monthly data through a list
# # # # 	##### daily_csv should be in the namelist.py
# # # # 	y1 = gv.Year1
# # # # 	y2 = gv.Year2
# # # # 	monthly_csv = gv.monthlycsv
# # # # 	with open(monthly_csv,'r') as f:
# # # # 		df_mary = f.readlines()

# # # # 	df_sub_uam = []
# # # # 	df_sub_vam = []
# # # # 	df_sub_hurm = []
# # # # 	df_sub_uad = []
# # # # 	df_sub_vad = []
                                                                        
# # # # 	for ia in df_mary:
# # # # 		#print(ia)
# # # # 		if 'u_component' in ia and 'daily' not in ia:
# # # # 			df_sub_uam.append(ia[:-1])
# # # # 		if 'u_component' in ia and 'dailymean' in ia:
# # # # 			df_sub_uad.append(ia[:-1])
# # # # 		if 'v_component' in ia and 'daily' not in ia:
# # # # 			df_sub_vam.append(ia[:-1])
# # # # 		if 'v_component' in ia and 'dailymean' in ia:
# # # # 			df_sub_vad.append(ia[:-1])
# # # # 		if 'relative_humidity' in ia:
# # # # 			df_sub_hurm.append(ia[:-1])

# # # # 	df_sub_uam = np.array(df_sub_uam)
# # # # 	df_sub_vam = np.array(df_sub_vam)
# # # # 	df_sub_hurm = np.array(df_sub_hurm)
# # # # 	df_sub_uad = np.array(df_sub_uad)
# # # # 	df_sub_vad = np.array(df_sub_vad)
# # # # 	modely1m = np.int_([df_sub_uam[i].split('/')[-1].split('_')[-1][:4] for i in range(df_sub_uam.shape[0])])
# # # # 	modely2m = np.int_([df_sub_uam[i].split('/')[-1].split('_')[-1].split('-')[-1][:4] for i in range(df_sub_uam.shape[0])])

# # # # 	##### reading daily data through Lamont URL
# # # # 	##### daily_csv should be in the namelist.py
# # # # 	modely1d = np.int_([df_sub_vad[i].split('/')[-1].split('_')[-2][:4] for i in range(df_sub_vad.shape[0])])
# # # # 	modely2d = np.int_([df_sub_vad[i].split('/')[-1].split('_')[-2][:4] for i in range(df_sub_vad.shape[0])])

# # # # 	t1 = time.time()

# # # # 	## TODO: should parallelize years 
# # # # 	arg_y1 = -1
# # # # 	neighbors=((0,1),(0,-1),(1,0),(-1,0),(1,1),(-1,1),(1,-1),(-1,-1),(0,2),(0,-2),(2,0),(-2,0))
# # # # 	for iy in range(gv.Year1,gv.Year2+1):
# # # # 		arg_yd = np.argwhere((modely1d<=iy)&(modely2d>=iy)).ravel()
# # # # 		arg_ym = np.argwhere((modely1m<=iy)&(modely2m>=iy)).ravel()[0]
# # # # 		#arg_ym0 = np.argwhere((modely1m<=iy)&(modely2m>=iy)).ravel().tolist()
# # # # 		#arg_ym1 = np.argwhere((modely1m<=iy+1)&(modely2m>=iy+1)).ravel().tolist()
# # # # 		#arg_ym2 = np.argwhere((modely1m<=iy-1)&(modely2m>=iy-1)).ravel().tolist()
# # # # 		#arg_ym0.extend(arg_ym1)
# # # # 		#arg_ym0.extend(arg_ym2)
# # # # 		#arg_ym = np.unique(np.array(arg_ym0))
# # # # 		if arg_yd[0] != arg_y1:
# # # # 									## following update was not here in original
# # # # 			arg_y1 = arg_yd[0]		## only re-open datasets when the daily file set changes.
# # # # 									## do we want this? TODO
# # # # 									## no speedup, don't know if there is an impact with multiple years
# # # # 			ds_uam = xr.open_dataset(df_sub_uam[arg_ym])
# # # # 			ds_vam = xr.open_dataset(df_sub_vam[arg_ym])
# # # # 			arg_p250m = np.argwhere(ds_uam.level.values==250.).ravel()[0]
# # # # 			arg_p850m = np.argwhere(ds_uam.level.values==850.).ravel()[0]

# # # # 			t = time.time()
# # # # 			ds_uad = xr.open_mfdataset(df_sub_uad[arg_yd])
# # # # 			ds_vad = xr.open_mfdataset(df_sub_vad[arg_yd])
# # # # 			print(f'  open_mfdataset={time.time()-t:.2f}')

# # # # 			arg_p250d = np.argwhere(ds_vad.level.values==250.).ravel()[0]
# # # # 			arg_p850d = np.argwhere(ds_vad.level.values==850.).ravel()[0]
# # # # 			xlong = ds_vam.longitude.values
# # # # 			xlat = ds_vam.latitude.values
# # # # 			dsvadtime = pd.to_datetime(np.arange(ds_vad.day.shape[0]),unit='D',origin=pd.Timestamp(str(iy)+'-01-01'))
		
# # # # 		###### This is a better way but xr.interP does no support chunk in the interpolation axis
# # # # 		#arg_tm = np.argwhere((ds_uam.time.dt.year.values==iy)|(ds_uam.time.dt.year.values==iy-1)|(ds_uam.time.dt.year.values==iy+1)).ravel()
# # # # 		arg_tm = np.argwhere((ds_uam.time.dt.year.values==iy)).ravel()
# # # # 		arg_td = np.argwhere((dsvadtime.year.values==iy)).ravel()
# # # # 		### cftime.DatetimeNoLeap  #### THIS IS REALLY for CESM2's calendar?? 
# # # # 		missing_value = 1e+20

# # # # 		date_daily = dsvadtime[arg_td]
# # # # 		date_monthly =ds_uam.time[arg_tm]

# # # # 		## updated from
# # # # 		### from /data0/clee/ERA5/pre
# # # # 		ndays_year = 0
# # # # 		m = []
# # # # 		for im in range(1,13):
# # # # 				m.append([monthrange(iy,im)[1]/2+ndays_year])
# # # # 				ndays_year = ndays_year+monthrange(iy,im)[1]
# # # # 		m = np.squeeze(np.int_(m)-1)

# # # # 		### on github
# # # # 		#m = np.argwhere(date_daily.day==15).ravel() ## roughly middle of the month

# # # # 		ua250m = ds_uam.u[arg_tm,arg_p250m].values
# # # # 		ua850m = ds_uam.u[arg_tm,arg_p850m].values
# # # # 		ua850m[ua850m!=ua850m] = 1e+20
# # # # 		ua850m = np.ma.masked_values(ua850m, missing_value)
# # # # 		ua850m = fillinNaN(ua850m,neighbors)
# # # # 		va250m = ds_vam.v[arg_tm,arg_p250m].values
# # # # 		va850m = ds_vam.v[arg_tm,arg_p850m].values
# # # # 		va850m[va850m!=va850m] = 1e+20
# # # # 		va850m = np.ma.masked_values(va850m, missing_value)
# # # # 		va850m = fillinNaN(va850m,neighbors)
# # # # 		d = np.arange(0,date_daily.shape[0])
# # # # 		f = interp1d(m,ua250m,bounds_error=False,fill_value="extrapolate",axis=0)
# # # # 		ua250md = f(d)
# # # # 		f = interp1d(m,va250m,bounds_error=False,fill_value="extrapolate",axis=0)
# # # # 		va250md = f(d)
# # # # 		f = interp1d(m,ua850m,bounds_error=False,fill_value="extrapolate",axis=0)
# # # # 		ua850md = f(d)
# # # # 		f = interp1d(m,va850m,bounds_error=False,fill_value="extrapolate",axis=0)
# # # # 		va850md = f(d)
	
				
# # # # 		t = time.time()
# # # # 		## original monthly covariance
# # # # 		##### now we are going to do the monthly cov.
# # # # 		covMatrix = np.zeros([10,12,va250md.shape[1],va250md.shape[2]])
# # # # 		for im in range(1,13):
# # # # 			arg_td = np.argwhere((dsvadtime.year==iy)&(dsvadtime.month==im)).ravel()
# # # # 			ua250d = ds_uad.u[arg_td,arg_p250d].values
# # # # 			ua850d = ds_uad.u[arg_td,arg_p850d].values
# # # # 			va250d = ds_vad.v[arg_td,arg_p250d].values
# # # # 			va850d = ds_vad.v[arg_td,arg_p850d].values
# # # # 			#missing_values = nc.variables['ua'].missing_value
# # # # 			missing_value = 1e+20
# # # # 			ua850d[ua850d!=ua850d] = 1e+20
# # # # 			va850d[va850d!=va850d] = 1e+20
# # # # 			va850d = np.ma.masked_values(va850d, missing_value)
# # # # 			ua850d = np.ma.masked_values(ua850d, missing_value)
# # # # 			ua850d = fillinNaN(ua850d,neighbors)
# # # # 			va850d = fillinNaN(va850d,neighbors)

# # # # 			arg_md = np.argwhere((dsvadtime[arg_td].year==iy)&(dsvadtime[arg_td].month==im)).ravel()
# # # # 			u250p = ua250d-ua250md[arg_md]
# # # # 			u250p2D = u250p.reshape([u250p.shape[0],u250p.shape[1]*u250p.shape[2]])
# # # # 			u850p = ua850d-ua850md[arg_md]
# # # # 			u850p2D = u850p.reshape([u250p.shape[0],u250p.shape[1]*u250p.shape[2]])
# # # # 			v250p = va250d-va250md[arg_md]
# # # # 			v250p2D = v250p.reshape([u250p.shape[0],u250p.shape[1]*u250p.shape[2]])
# # # # 			v850p = va850d-va850md[arg_md]
# # # # 			v850p2D = v850p.reshape([u250p.shape[0],u250p.shape[1]*u250p.shape[2]])
			
# # # # 			### TODO:
# # # # 			### probably should replace eval with direct indexing 
# # # # 			var = ['u250p2D','v250p2D','u850p2D','v850p2D']
# # # # 			count = 0
# # # # 			for iv in range(len(var)):
# # # # 				vname = var[iv]
# # # # 				for iiv in range(iv,len(var),1):
# # # # 					vname1 = var[iiv]
# # # # 					covMatrix[count,im-1,:,:] = \
# # # # 						my_cov(eval(vname),eval(vname1),0).reshape([u250p.shape[1],u250p.shape[2]])
# # # # 						#np.hstack([np.cov(eval(vname)[:,igrid],eval(vname1)[:,igrid])[0,1] \
# # # # 						#for igrid in range(u250p2D.shape[1])]).reshape([u250p.shape[1],u250p.shape[2]])
# # # # 					count += 1
# # # # 			print(iy, im, count, vname, vname1)

# # # # 		print(f'  monthly covariance={time.time()-t:.2f}')

# # # # 		t2 = time.time()
# # # # 		createNetCDF(covMatrix,iy,xlong,xlat)

# # # # 		del covMatrix,u250p, u250p2D, u850p, u850p2D, v250p, v250p2D, v850p, v850p2D  

# # # # 		t3 = time.time()

# # # # 		gc.collect()

# # # # 		print(f'calWind run times: import={t1-t0:.2f}  big for loop={t2-t1:.2f}  createNetCDF={t3-t2:.2f}')


