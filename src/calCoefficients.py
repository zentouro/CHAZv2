### #!/usr/bin/env python
###############################
## Generate coefficient data ##
## and global predictors     ##
## for CHAZ from best tracks ##
## and ERA5 data             ##
###############################



## necesarry imports and helper functions 

import gc
import time
import copy
import pickle


import numpy as np
from numpy import nanmean
import pandas as pd
import xarray as xr
import dask.array as da
import statsmodels.api as sm

from datetime import datetime,timedelta
from scipy.io import loadmat, netcdf_file
from netCDF4 import Dataset

import src.module_riskModel as mrisk
import Namelist as gv

from tools.util import int2str,date_interpolation



class Vregression1():
    """
    creates regression object
    """
    def __init__(self,n1,n2):
        self.y=np.empty([n1],dtype=float)*float('nan')
        self.x=np.empty([n1,n2],dtype=float)*float('nan')



# no special Time branch
class mybt:
    def __init__(self, bt_array):
        for iv in bt_array.keys():
            setattr(self, iv, bt_array[iv])      

## original from readbst.py
# class mybt:
# 	def __init__(self,bt_array):
# 		for iv in bt_array.keys():
# 			if 'Time' not in iv:
# 				setattr(self, iv, bt_array[iv])
# 			else:
# 				start = pd.Timestamp('1800-01-01').to_pydatetime()
# 				setattr(self, iv, start+np.int_(bt_array[iv]*24)*timedelta(hours=1))


def normalizeTransform(X,Y):
   """
   Find mean and standard deviation of X and Y
   """
   meanX = []
   stdX = []
   meanX.append(0.0)
   stdX.append(0.0)
   meanY = np.mean(Y)
   stdY = np.std(Y)
   Y = (Y - meanY)/stdY
   for ix in range(1,X.shape[1],1):
      meanX.append(np.mean(X[:,ix]))
      stdX.append(np.std(X[:,ix]))
      X[:,ix] = (X[:,ix] - meanX[ix])/stdX[ix]

   return X,meanX,stdX,Y,meanY,stdY

def get_landmask(filename):
    """
    This function reads landmask.nc. 
    read 0.25degree landmask.nc  -- updating to 0.125
    output:
        lon: longitude, 1D
        lat: latitude, 1D
        landmask:2D
  
    """
    f = netcdf_file(filename)
    lon = f.variables['lon'][:]
    lat = f.variables['lat'][:]
    landmask = f.variables['landmask'][:,:]
    f.close()

    return lon, lat, landmask

def getSpeedDir(iiS,block_id=None):
    ## iS was reporting as a single item array [TK]
    iS = np.int_(iiS.mean(keepdims=True))[0]
    #iS = np.int(iiS.mean(keepdims=True))
    if (bt['StormLat'][:,iS]==bt['StormLat'][:,iS]).any():
        it1 = np.argwhere(bt['StormLat'][:,iS]==bt['StormLat'][:,iS])[0,0]
        it2 = np.argwhere(bt['StormLat'][:,iS]==bt['StormLat'][:,iS])[-1,0]
        if it2 - it1 >=2:
           bt['trDir'][it1:it2,iS],bt['trSpeed'][it1:it2,iS]=\
             getStormTranslation(bt['StormLon'][it1:it2,iS],\
             bt['StormLat'][it1:it2,iS],bt['Time'][it1:it2,iS])
    return iS


def getStormTranslation(lon,lat,time):

    er = 6371.0 #km
    delt = (time[1:]-time[:-1])*24*60*60
    dlon = lon[1:]-lon[:-1]
    dlat = lat[1:]-lat[:-1]
    speed = np.zeros(lon.shape[0],dtype=float)+float('nan')
    sdir = np.zeros(lon.shape[0],dtype=float)+float('nan')
    londis = 2*np.pi*er*np.cos(lat[0:-1]/180*np.pi)/360
    dx = londis*(dlon)
    dy = 110*(dlat)
    distance = np.sqrt(dx*dx+dy*dy) #km
    sdir[:-1]=np.arctan2(dlat,dlon)
    speed[:-1]=distance*1000./delt #m/s

    return sdir,speed


## from getMeanStd.py - updated landmask
def get_predictors(iiS, ty1, ty2, block_id=None):
    ## iS was reporting as a single item array [TK]
    iS = np.int_(iiS.mean(keepdims=True))[0]
    #iS = np.int(iiS.mean(keepdims=True))
    #predictors=['StormMwspd','dVdt','trSpeed','dPIwspd','SHRD','rhMean','dPIwspd2','dPIwspd3','dVdt2','landmaskMean']
    
    ### for training, only want to use years 1980
    if ( (int(bt['StormYear'][iS]) >= ty1) & (int(bt['StormYear'][iS]) < ty2) ):
        if bt['StormYear'][iS]>1980:
            fileName = gv.pre_path+str(bt['StormYear'][iS])+'_'+gv.ENS+'.nc'
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

            ## TODO: use same landmask updates? 
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

            for it in range(0,bt['StormLon'][:,iS].shape[0],1):	
                if bt['Time'][it,iS] > 0:
                    #distance = np.empty(xxlong.shape,dtype=float)
                    #er = 6371.0 #km
                    #londis = 2*np.pi*er*np.cos(xxlat/180*np.pi)/360
                    dx = londis*(xxlong - bt['StormLon'][it,iS])
                    dy = 110 * (xxlat - bt['StormLat'][it,iS])
                    distance = np.sqrt(dx*dx+dy*dy)
                    (j0,i0) = np.unravel_index(np.argmin(distance),distance.shape)

                    var,radius1,radius2 =  date_interpolation(datetime(1800,1,1)+timedelta(days = bt['Time'][it,iS]),PIVmax),0,500
                    #var,radius1,radius2 =  date_interpolation(datetime(1800,01,01)+timedelta(days = bt['Time'][it,iS]),PIVmax),0,500
                    bt['PIwspdMean'][it,iS] = np.mean(var[(distance<=radius2) & (distance>=radius1) & (var==var)])
                    bt['PIwspd'][it,iS] = var[j0,i0]

                    var,radius1,radius2 =  date_interpolation(datetime(1800,1,1)+timedelta(days = bt['Time'][it,iS]),u),200,800
                    #var,radius1,radius2 =  date_interpolation(datetime(1800,01,01)+timedelta(days = bt['Time'][it,iS]),u),200,800
                    bt['UShearMean'][it,iS] = np.mean(var[(distance<=radius2) & (distance>=radius1) & (var==var)])
                    bt['UShear'][it,iS] = var[j0,i0]

                    var,radius1,radius2 =  date_interpolation(datetime(1800,1,1)+timedelta(days = bt['Time'][it,iS]),v),200,800
                    #var,radius1,radius2 =  date_interpolation(datetime(1800,01,01)+timedelta(days = bt['Time'][it,iS]),v),200,800
                    bt['VShearMean'][it,iS] = np.mean(var[(distance<=radius2) & (distance>=radius1) & (var==var)])
                    bt['VShear'][it,iS] = var[j0,i0]

                    var,radius1,radius2 =  date_interpolation(datetime(1800,1,1)+timedelta(days = bt['Time'][it,iS]),meanrh),200,800
                    #var,radius1,radius2 =  date_interpolation(datetime(1800,01,01)+timedelta(days = bt['Time'][it,iS]),meanrh),200,800
                    bt['rhMean'][it,iS] = np.mean(var[(distance<=radius2) & (distance>=radius1) & (var==var)])
                    bt['rh'][it,iS] = var[j0,i0]

                    # var,radius1,radius2 = copy.copy(ldmask),0,300
                    # var[var==0] = -1.
                    # var[var==3] = 0.0
                    # bt['landmaskMean'][it,iS] = np.mean(var[(distance<=radius2) & (distance>=radius1) & (var==var)])
                    # bt['landmask'][it,iS] = var[j0,i0] 

                    var,radius1,radius2 = copy.copy(gv.ldldmask),0,300
                    var[var==0] = -1.
                    var[var==3] = 0.0
        
                    ### recalculating the ix,iy landmask ####  
                    # distance = np.empty(ldxxlong.shape,dtype=float)
                    # londis = 2*np.pi*er*np.cos(ldxxlat/180*np.pi)/360
                    dx = londis75*(ldxxlong - bt['StormLon'][it,iS])
                    dy = 110 * (ldxxlat - bt['StormLat'][it,iS])  
                    distance75 = np.sqrt(dx*dx+dy*dy)  
                    (j0,i0) = np.unravel_index(np.argmin(distance75),distance75.shape)
                    ###########################################  
        
                    bt['landmaskMean'][it,iS] = np.mean(var[(distance75<=radius2) & (distance75>=radius1) & (var==var)])
                    bt['landmask'][it,iS] = var[j0,i0] 

    return iS


def calMultiRegression_coef_mean_water(bt,leadingTime,ty1,ty2,predictors,fstType):
   """
   calculate mean coefficients for multiple regression for water
   bt: best track data
   ty1: year1
   ty2: year2
   predictors: predictors
   fstType: fst variable type
   """
   import copy

   lT = np.int_(leadingTime/6)
   n1 = bt.StormMwspd.shape[0]*bt.StormMwspd.shape[1]
   n2 = len(predictors)
   varRegress = Vregression1(n1,n2)
   #varRegress1 = Vregression1(n1,n2)
   
   ### index for errors/residuals
   itIdx = np.full(n1, -1, dtype=int)
   isIdx = np.full(n1, -1, dtype=int)

   TimeDepends = ['dThetaEMean','T200Mean',
                  'rhMean','rh500_300Mean',
                  'div200Mean','T100Mean',
                  'dThetaEsMean'
                  ] 

   iS = bt.StormYear.tolist().index(ty1) 
   count1 = 0

   ## debugging
   # predictor_names = ['StormMwspd','dVdt','trSpeed','dPIwspd','SHRD','rhMean','dPIwspd2','dPIwspd3','dVdt2']
   # for i, name in enumerate(predictor_names):
   #    col = varRegress.x[:count1, i]
   #    print(f'{name}: {np.isnan(col).sum()} / {count1} NaN  (min={np.nanmin(col) if not np.all(np.isnan(col)) else "all-nan"}, max={np.nanmax(col) if not np.all(np.isnan(col)) else "all-nan"})')

   while True:
      if ( (int(bt.StormYear[iS]) >= ty1) & (int(bt.StormYear[iS]) < ty2) ):
         for it in range(lT,bt.StormLon[:,iS].shape[0],1):
            #### if the starting or destination point has more than 50% land over 300km radius
            if((bt.StormLon[it,iS]==bt.StormLon[it,iS])&((bt.landmaskMean[it,iS]<= -0.5)&(bt.landmaskMean[it-lT,iS]<= -0.5))):
               if (fstType == 'Vmax'):
                  varRegress.y[count1]=bt.StormMwspd[it,iS] 	
               elif (fstType == 'dVmax'):
                  varRegress.y[count1]=bt.StormMwspd[it,iS]-bt.StormMwspd[it-lT,iS] 	
            count2 = 0
            for var in predictors:
               if var in 'SHRD':
                  varRegress.x[count1,count2] = nanmean(np.sqrt(bt.UShearMean[it-lT:it+1,iS]**2+bt.VShearMean[it-lT:it+1,iS]**2))
               elif var in 'SHRD_I':
                  varRegress.x[count1,count2] = np.sqrt(bt.UShearMean[it-lT,iS]**2+bt.VShearMean[it-lT,iS]**2)
               elif var in 'dPIwspd':
                  varRegress.x[count1,count2] = nanmean(bt.PIwspdMean[it-lT:it+1,iS])-bt.StormMwspd[it-lT,iS]	
               elif var in 'dPIslp':
                  varRegress.x[count1,count2] = nanmean(bt.PIslp[it-lT:it+1,iS])-bt.StormMslp[it-lT,iS]	
               elif var in TimeDepends:
                  varRegress.x[count1,count2] = nanmean(getattr(bt,var)[it-lT:it+1,iS])
               elif var in 'landmaskMean':
                  varRegress.x[count1,count2] = getattr(bt,var)[it,iS]-getattr(bt,var)[it-lT,iS]
               elif var in 'StormMwspd':
                  varRegress.x[count1,count2] = (getattr(bt,'StormMwspd')[it-lT,iS])
               elif var in 'StormMwspd2':
                  varRegress.x[count1,count2] = (getattr(bt,'StormMwspd')[it-lT,iS])**2
               elif var in 'StormMwspd3':
                  varRegress.x[count1,count2] = (getattr(bt,'StormMwspd')[it-lT,iS])**3
               elif var in 'MPI':
                  varRegress.x[count1,count2] = nanmean(bt.PIwspdMean[it-lT:it+1,iS])
               elif var in 'MPI2':
                  varRegress.x[count1,count2] = nanmean(bt.PIwspdMean[it-lT:it+1,iS])**2
               elif var in 'dPIwspd2':
                  varRegress.x[count1,count2] = (nanmean(bt.PIwspdMean[it-lT:it+1,iS])-bt.StormMwspd[it-lT,iS])**2
               elif var in 'dPIwspd3':
                  varRegress.x[count1,count2] = (nanmean(bt.PIwspdMean[it-lT:it+1,iS])-bt.StormMwspd[it-lT,iS])**3
               elif 'dVdt2' in var:
                  varRegress.x[count1,count2] = (getattr(bt,'dVdt')[it-lT,iS])**2 
               elif 'dVdt3' in var:
                  varRegress.x[count1,count2] = (getattr(bt,'dVdt')[it-lT,iS])**3 
               elif var in 'pre_V0':
                  if it-lT >= 2:
                     varRegress.x[count1,count2] = getattr(bt,'StormMwspd')[it-lT-2,iS]
                  else:
                     varRegress.x[count1,count2] = getattr(bt,'StormMwspd')[it,iS]
               elif var in 'pre_V02':
                  if it-lT >= 2:
                        varRegress.x[count1,count2] = (getattr(bt,'StormMwspd')[it-lT-2,iS])**2
                  else:
                        varRegress.x[count1,count2] = (getattr(bt,'StormMwspd')[it,iS])**2
               else:
                  varRegress.x[count1,count2] = getattr(bt,var)[it-lT,iS]
               count2 += 1

            ## keep track of row/column for residuals/errors
            itIdx[count1] = it
            isIdx[count1] = iS
            count1 += 1 
 
      iS += 1
      #print(iS)
      #print(bt.StormYear.shape[0]-1)
      if iS > bt.StormYear.shape[0]-1:break
      
   #print(varRegress1.x[0,:])

   a = 1
   for iv in range(0,n2,1):
      a = a*varRegress.x[:,iv] 	
   a = a*varRegress.y

   # print('rows written (count1):', count1)
   # print('valid rows after NaN filter:', np.sum(a == a))
   # print('rows with any-NaN y:', np.sum(np.isnan(varRegress.y[:count1])))
   # print('rows with any-NaN x:', np.sum(np.isnan(varRegress.x[:count1, :]).any(axis=1)))
   # print('landmaskMean range this window:', np.nanmin(bt.landmaskMean), np.nanmax(bt.landmaskMean))

   #print(varRegress.x)

   # predictor_names = ['StormMwspd','dVdt','trSpeed','dPIwspd','SHRD','rhMean','dPIwspd2','dPIwspd3','dVdt2']
   # for i, name in enumerate(predictor_names):
   #    col = varRegress.x[:count1, i]
   #    print(f'{name}: {np.isnan(col).sum()} / {count1} NaN  (min={np.nanmin(col) if not np.all(np.isnan(col)) else "all-nan"}, max={np.nanmax(col) if not np.all(np.isnan(col)) else "all-nan"})')


   X = sm.add_constant(varRegress.x[a==a,:], prepend = 'True')
   Y = varRegress.y[a==a]
   X,meanX,stdX,Y,meanY,stdY = normalizeTransform(X,Y) 
   results = sm.OLS(Y,X).fit()

   ## keeping track of residuals/errors
   rows  = np.where(a == a)[0]                ## same mask used to build X, Y
   resid = np.asarray(results.resid) * stdY   ## back to physical units (m/s)
   #pred  = np.asarray(results.fittedvalues) * stdY + meanY

   errs = np.full(bt.StormMwspd.shape, np.nan)
   errs[itIdx[rows], isIdx[rows]] = resid

   return results, meanX,meanY,stdX,stdY, errs


def calMultiRegression_coef_mean_land(bt,leadingTime,ty1,ty2,predictors,fstType):
   """
   calculate mean coefficients for multiple regression on land
   bt: best track data
   ty1: year1
   ty2: year2
   predictors: predictors
   fstType: fst variable type
   """
   import copy

   lT = np.int_(leadingTime/6)
   n1 = bt.StormMslp.shape[0]*bt.StormMslp.shape[1]
   n2 = len(predictors)
   varRegress = Vregression1(n1,n2)
   # = Vregression1(n1,n2)

   ### index for errors/residuals
   itIdx = np.full(n1, -1, dtype=int)
   isIdx = np.full(n1, -1, dtype=int)

   TimeDepends = ['dThetaEMean','T200Mean','rhMean','rh500_300Mean','div200Mean','T100Mean','dThetaEsMean'] 


   iS = bt.StormYear.tolist().index(ty1) 
   count1 = 0
   while True:
      if ( (int(bt.StormYear[iS]) >= ty1) & (int(bt.StormYear[iS]) < ty2) ):
         for it in range(lT,bt.StormLon[:,iS].shape[0],1):
            #### if the starting or destination point has more than 50% land over 300km radius
            if((bt.StormLon[it,iS]==bt.StormLon[it,iS])&((bt.landmaskMean[it,iS]> -0.5)|(bt.landmaskMean[it-lT,iS]> -0.5))):
               if (fstType == 'Vmax'):
                  varRegress.y[count1]=bt.StormMwspd[it,iS] 	
               elif (fstType == 'dVmax'):
                  varRegress.y[count1]=bt.StormMwspd[it,iS]-bt.StormMwspd[it-lT,iS] 	
            count2 = 0
            for var in predictors:
               if var in 'SHRD':
                  varRegress.x[count1,count2] = nanmean(np.sqrt(bt.UShearMean[it-lT:it+1,iS]**2+bt.VShearMean[it-lT:it+1,iS]**2))
               elif var in 'SHRD_I':
                  varRegress.x[count1,count2] = np.sqrt(bt.UShearMean[it-lT,iS]**2+bt.VShearMean[it-lT,iS]**2)
               elif var in 'dPIwspd':
                  varRegress.x[count1,count2] = nanmean(bt.PIwspdMean[it-lT:it+1,iS])-bt.StormMwspd[it-lT,iS]	
               elif var in 'dPIslp':
                  varRegress.x[count1,count2] = nanmean(bt.PIslp[it-lT:it+1,iS])-bt.StormMslp[it-lT,iS]	
               elif var in TimeDepends:
                  varRegress.x[count1,count2] = nanmean(getattr(bt,var)[it-lT:it+1,iS])
               elif var in 'landmaskMean':
                  varRegress.x[count1,count2] = getattr(bt,var)[it,iS]-getattr(bt,var)[it-lT,iS]
               elif var in 'StormMwspd':
                  varRegress.x[count1,count2] = (getattr(bt,'StormMwspd')[it-lT,iS])
               elif var in 'StormMwspd2':
                  varRegress.x[count1,count2] = (getattr(bt,'StormMwspd')[it-lT,iS])**2
               elif var in 'StormMwspd3':
                  varRegress.x[count1,count2] = (getattr(bt,'StormMwspd')[it-lT,iS])**3
               elif var in 'MPI':
                  varRegress.x[count1,count2] = nanmean(bt.PIwspdMean[it-lT:it+1,iS])
               elif var in 'MPI2':
                  varRegress.x[count1,count2] = nanmean(bt.PIwspdMean[it-lT:it+1,iS])**2
               elif var in 'dPIwspd2':
                  varRegress.x[count1,count2] = (nanmean(bt.PIwspdMean[it-lT:it+1,iS])-bt.StormMwspd[it-lT,iS])**2
               elif var in 'dPIwspd3':
                  varRegress.x[count1,count2] = (nanmean(bt.PIwspdMean[it-lT:it+1,iS])-bt.StormMwspd[it-lT,iS])**3
               elif 'dVdt2' in var:
                  varRegress.x[count1,count2] = (getattr(bt,'dVdt')[it-lT,iS])**2 
               elif 'dVdt3' in var:
                  varRegress.x[count1,count2] = (getattr(bt,'dVdt')[it-lT,iS])**3 
               elif var in 'pre_V0':
                  if it-lT >= 2:
                     varRegress.x[count1,count2] = getattr(bt,'StormMwspd')[it-lT-2,iS]
                  else:
                     varRegress.x[count1,count2] = getattr(bt,'StormMwspd')[it,iS]
               elif var in 'pre_V02':
                  if it-lT >= 2:
                     varRegress.x[count1,count2] = (getattr(bt,'StormMwspd')[it-lT-2,iS])**2
                  else:
                     varRegress.x[count1,count2] = (getattr(bt,'StormMwspd')[it,iS])**2
               else:
                  varRegress.x[count1,count2] = getattr(bt,var)[it-lT,iS]

               count2 += 1

            ## keep track of row/column for residuals/errors
            itIdx[count1] = it
            isIdx[count1] = iS
            count1 += 1

 
      iS += 1
      if iS > bt.StormYear.shape[0]-1:break
	
   #print(varRegress1.x[0,:])

   a = 1
   for iv in range(0,n2,1):
      a = a*varRegress.x[:,iv] 	
   a = a*varRegress.y

   X = sm.add_constant(varRegress.x[a==a,:],prepend = 'True')
   Y = varRegress.y[a==a]
   X,meanX,stdX,Y,meanY,stdY = normalizeTransform(X,Y) 
   results = sm.OLS(Y,X).fit()

   ## keeping track of residuals/errors
   rows  = np.where(a == a)[0]                ## same mask used to build X, Y
   resid = np.asarray(results.resid) * stdY   ## back to physical units (m/s)
   #pred  = np.asarray(results.fittedvalues) * stdY + meanY

   errs = np.full(bt.StormMwspd.shape, np.nan)
   errs[itIdx[rows], isIdx[rows]] = resid

   return results, meanX,meanY,stdX,stdY, errs

def calCoefficient_water_guess(ty1, ty2, ih):
   """
   calculate first guess coefficient from OLS
   """
   #  from tools.regression4 import calMultiRegression_coef_mean_water
   #  from tools.regression4 import calMultiRegression_coef_mean_land

   print('starting water')
   ### calculatae coefficients
   predictors = ['StormMwspd','dVdt','trSpeed','dPIwspd','SHRD','rhMean','dPIwspd2','dPIwspd3','dVdt2']
   result_w,meanX_w,meanY_w,stdX_w,stdY_w,errs_w = \
      calMultiRegression_coef_mean_water(bt,ih,\
      ty1,ty2,predictors,'dVmax')

   print('starting land')
   predictors=['StormMwspd','dVdt','trSpeed','dPIwspd','SHRD','rhMean','dPIwspd2','dPIwspd3','dVdt2','landmaskMean']
   result_l,meanX_l,meanY_l,stdX_l,stdY_l,errs_l = \
      calMultiRegression_coef_mean_land(bt,ih,\
      ty1,ty2,predictors,'dVmax')

   # combine water and land residuals 
   errs = np.where(np.isnan(errs_w), errs_l, errs_w)
   bt.errors = errs        # attribute access, since you use bt.StormMwspd etc.

   return(result_w,meanX_w,meanY_w,stdX_w,stdY_w,
          result_l,meanX_l,meanY_l,stdX_l,stdY_l,
          errs)


def bt_to_xr(bt, template):
    data_vars, missing = {}, []
    for name, tvar in template.data_vars.items():
        if not hasattr(bt, name):
            missing.append(name)
            continue

        v = np.asarray(getattr(bt, name))

        # match template shape (squeeze (1,N) arrays, fix transposed 2-D arrays)
        if v.shape != tvar.shape:
            v = np.squeeze(v)
        if v.ndim == 2 and v.shape == tvar.shape[::-1]:
            v = v.T
        if v.shape != tvar.shape:
            raise ValueError(f"{name}: bt shape {v.shape} != template {tvar.shape}")

        ## if mybt converts time to datetime
        # if 'Time' in name:
        #     # object array of datetimes -> datetime64, works for 1-D or 2-D
        #     v = pd.to_datetime(v.ravel()).values.reshape(v.shape)
        
        elif v.dtype != tvar.dtype:
            v = v.astype(tvar.dtype)      # e.g. keeps basin as |S2

        data_vars[name] = (tvar.dims, v, tvar.attrs)

    print("not on bt, skipped:", missing)
    print("on bt, not in template:", set(vars(bt)) - set(template.data_vars))
    return xr.Dataset(data_vars, attrs=template.attrs)


## export result_l and result_w to ipath (as devined in Namelist.py)
def save_params(result, path):
    xr.Dataset({'params': ('d1', np.asarray(result.params))}).to_netcdf(path)


def run_calCoefficients():

    ipath = gv.ipath
    iy1, iy2 = 1981, 2025

    global ldmask
    landmaskfile = 'input/landmask.nc'
    llon, llat, dummy = get_landmask(landmaskfile)
    #ldmask = dummy[-12::-24,::24]  #<- original, should this update to
    ldmask = dummy[::-24,  ::24]  ## update landmask to higher resolution? 

    if True:
        ## pull in best tracks from resize_basinData
        global bt
        bt = mrisk.resize_basinData()

        bt['StormYear'] = np.int16(bt['StormYear'])

        time1 = time.time()
        nS = bt['div200'].shape[1]
        diS = da.from_array(np.arange(0,nS,1).astype(dtype=np.int32),chunks=(1,))
        niS = np.arange(0,nS,1).astype(dtype=np.int32)
        new = da.map_blocks(getSpeedDir,diS,chunks=(1,),dtype=diS.dtype)
        new.compute()
        print('done translation speed', np.round(time.time()-time1, 3), 'seconds')
        del new
        gc.collect()

        ### calculate predictors ###
        for iS in niS: 
            ## updates bt each time
            get_predictors(np.array([iS,iS]), iy1, iy2)


        #time1 = time.time()
        ## this is just as slow as the for loop i think 
        # new = da.map_blocks(get_predictors,diS,chunks=(1,1))
        # a = new.compute()

        print('done calPredictors at ', np.round((time.time()-time1)/60, 3), 'minutes')

        gc.collect()
        bt1 = copy.deepcopy(bt)
        del bt
        bt = mybt(bt1)
        with open(ipath+'trackPredictorsbt_obs.pik','wb')as f:
                pickle.dump(bt,f)
        #with open(ipath+'trackPredictorsbt_obs.pik','w+')as f:
        #        pickle.dump(bt,f)
        f.close()
    else:
        with open(ipath+'trackPredictorsbt_obs.pik','r')as f:
            bt = pickle.load(f)
        f.close()

    print('done pickling at ', np.round((time.time()-time1)/60, 3), 'minutes')

    print('start calCoefficient_water_guess at ', np.round((time.time()-time1)/60, 3), 'minutes')

    result_w,meanX_w,meanY_w,stdX_w,stdY_w,result_l,meanX_l,meanY_l,stdX_l,stdY_l,errs = calCoefficient_water_guess(iy1,iy2+1,12)

    print('done calCoefficient_water_guess at ', np.round((time.time()-time1)/60, 3), 'minutes')

    meanX_w = np.array(meanX_w)
    stdX_w = np.array(stdX_w)
    meanY_w = np.array(meanY_w)
    stdY_w = np.array(stdY_w)
    meanX_l = np.array(meanX_l)
    stdX_l = np.array(stdX_l)
    meanY_l = np.array(meanY_l)
    stdY_l = np.array(stdY_l)

    d2 = np.empty(meanX_w.shape[0])
    d3 = np.empty(stdX_w.shape[0])
    d7 = np.empty(meanX_l.shape[0])
    d8 = np.empty(stdX_l.shape[0])


    ds = xr.Dataset({
    'meanX_w': xr.DataArray(
                    data = meanX_w,
                    dims = ['d2']),


    'stdX_w': xr.DataArray(
                    data = stdX_w,
                    dims = ['d3']),


    'meanY_w': xr.DataArray(
                    data = meanY_w),


    'stdY_w': xr.DataArray(
                    data = stdY_w),

    'meanX_l': xr.DataArray(
                    data = meanX_l,
                    dims = ['d7']),


    'stdX_l': xr.DataArray(
                    data = stdX_l,
                    dims = ['d8']),


    'meanY_l': xr.DataArray(
                    data = meanY_l),


    'stdY_l': xr.DataArray(
                    data = stdY_l),

            }
        )

    print('ds created', (time.time()-time1)/60, 'minutes')


    ## export coefficient_mean_std.nc to ipath (as defined in Namelist.py)
    filename = ipath+'coefficient_meanstd.nc'
    ds.to_netcdf(filename)

    save_params(result_l, ipath+'result_l.nc')
    save_params(result_w, ipath+'result_w.nc')

    ##TODO: Fix this hardcoded silliness
    bt_global = xr.open_dataset("/home/miriamn/CHAZ/CHAZvectorized/input/updating/bt_global.nc").load()

    ds_global_predictors = bt_to_xr(bt, bt_global)
    ds_global_predictors.to_netcdf(gv.ipath+'bt_global_predictors.nc')

