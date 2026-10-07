#!/usr/bin/env python

### suppresses the following, uncomment to see if new ones have appeared 
# RuntimeWarning: Cannot close a netcdf_file opened with mmap=True, when netcdf_variables or 
# arrays referring to its data still exist. All data arrays obtained from such files refer 
# directly to data on disk, and must be copied before the file can be cleanly closed. 
# (See netcdf_file docstring for more information on mmap.)
import warnings
warnings.filterwarnings('ignore')  # Suppress all warnings

from scipy.io import loadmat, netcdf_file

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

### Experiment settings
Model = 'ERA5'
ENS = 'r1i1p1f1'                   # Designed for CMIP6 - but useful for file naming
TCGIinput = 'TCGI_CRH_PI'          # or "TCGI_SD_RI" or 'TCGI_CRH'
CHAZ_ENS_0 = 0           
CHAZ_ENS = 3                       # Number of ensemble realizations.  
CHAZ_Int_ENS = 40                  # Number of intensity realizations

### CHAZ parameters


# file names of model variables 
monthlycsv = 'model-data-fnames.txt'
dailycsv = 'model-data-fnames.txt'

uBeta = -1.5
vBeta = 2.0
survivalrate = 0.78
seedN = 1000                                 
landmaskfile = 'input/landmask.nc'      
ipath = 'input/'                        ## ipath contains input data from observations (best tracks for all the basins from IBTrACS)
opath = 'input/bt_global_predictors.nc' ## and a global best track with intial predictors 

## Preprocessing output path AND CHAZ import path
pre_path = 'pre/'


## CHAZ output path
output_path = 'output/'


## Sample time frame, with full ERA5 and input data
## CHAZ can be run from 1950 to present
Year1 = 2020
Year2 = 2025

### Defining landmask
llon, llat,lldmask = get_landmask(landmaskfile)
ldmask = lldmask[::-24,  ::24]          ## 2º
ldldmask = lldmask[::-9, ::9]           ## .75º [used in module_GenBamPred.bam and .get_predictors]
ldlon = llon[::9]
ldlat = llat[::9]
lldmask = lldmask[::-1,:]               ## flips latitudes so they are in the right order



####################################################
#### Running CHAZv2                             ####
####                                            ####
####                                            ####
#####################################################

overwrite = True      ## select True to overwrite output                         


## for replicability, can define a random seed
random_seed = 42
local_random = False        

## Logging not yet implemented
## output progress and timing logs 
#log_path = '/logs/'
## log_level = TK 


### Partially implemented, room for improvement and speedup
# debugging/verbose
debugging = False     ## select True to include debug print statements and file saves     --  # partially implemented
quiet = False         ## select True to suppress print statements while running           --  # partially implemented


#####################################################
## Preprocesses                                   ###
## ignore variables when runPreprocess = False    ###
#####################################################
runPreprocess = True 
calWind = True 
calpreProcess = True 
calA = True

#####################################################
## CHAZ                                           ###
## ignore variables when runCHAZ = False          ###
#####################################################
runCHAZ = True
### genesis 
calGen = True   
### track
calBam =  True
### intensiy
calInt = True 

