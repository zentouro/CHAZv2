** THIS IS A TEMPORARY REPOSITORY FOR DEMONSTRATION PURPOSES ONLY **
Please see the main CHAZ repository here: https://github.com/cl3225/CHAZ

# CHAZ (Columbia Tropical Cyclone Hazard Model)

## I. Overview 

The Columbia HAZard model (CHAZ) is a statistical-dynamical downscaling model for estimating tropical cyclone hazards. The CHAZ model consists of three primary components for describing tropical cyclone activity from genesis to lysis. These components are genesis, track, and intensity modules. The genesis module uses Tropical Cyclone Genesis Index (TCGI, developed by [Tippett et al, 2011](https://doi.org/10.1175/2010JCLI3811.1) to estimate the seeding rate of the storm precursors. The seeds are then passed to a  Beta-Advection Model (BAM) that moves the storm forward with giving environmental sterring flow. The intensity module then evolves storms beyond genesis using an Autoregressive Model that involves a deterministic and a stochastic forcing elements. The underlying science of the CHAZ model can be found at [Lee et al. 2018](https://doi.org/10.1002/2017MS001186)).

Generally speaking, there are two primary steps for running CHAZ: preprocessing and downscaling. Preprocessing includes collecting global model data, calculating PI, TCGI, wind covariance matrix, putting the data into a standard format for downscaling calculation, etc. Preprocessing codes are model-dependent, meaning that you will need to modify them when switch global models. Codes for conducting downscaling are insensitive to what global model you are using. Note, this code package does not calculate PI or TCGI. 

## II. Getting Started with CHAZ

### Environment
CHAZv2 uses Python 3. Ensure a package manager (e.g. Anaconda, conda, mamba) and Python 3 are installed on your machine. 

Use your package manager of choice to create a new environment from the supplied yaml file. 

`conda create env --file tools/chaz_env.yaml`

Then activate that environment:

`conda activate chaz`

### CHAZ structure:
```bash
├── input
│   ├── bt_*.nc
│   ├── bt_global_predictors.nc
│   ├── coefficient_meanstd.nc
├── output
│   ├── ERA5_*_ens***.nc
│   ├── trackPredictorsbt*_ens*.pik
├── pre
│   ├── YYYY*_r1i1p1f1.nc
│   ├── A_*YYYYMM.nc.nc
│   ├── Cov*YYYY.nc
│   ├── PI_*YYYY.mat
│   ├── TCGI_CRH_PI_*YYYY.mat
├── preprocess_ERA5
│   ├── convert6hourly-daily.py
│   ├── era5_download_daily.py
│   ├── era5_download.py
├── src
│   ├── bt_gen.py
│   ├── calA.py
│   ├── calCoefficients.py
│   ├── caldesto_track_lysis_v2.py
│   ├── calWindCov_v2.py
│   ├── module_genBamPred_v2.py
│   ├── module_riskModel.py
│   ├── preprocess_v2.py
├── tools
│   ├── chaz_env.yaml
│   ├── module_stochastic_v2.py
│   ├── util.py
├── CHAZ.py
├── Namelist.py
├── README.md
└── .gitignore
```

In CHAZ, Namelist.py defines all global variables such as the source of the global model forcing, the ensemble members of that global model, the version of genesis module, numbers of track and intensity ensemble members, and whether you are doing CHAZ-pre or CHAZ downscaling. In theory, Namelist.py is the only file you need to modify. CHAZ.py reads in Namelist.py and is the script that calls for all the modules/subroutines. Below we describe input variables that need to be modified when conducting CHAZ-pre and CHAZ downscaling.  

`Namelist.py`: python file containing global variables modified for CHAZ preprocessing and CHAZ downscaling

`CHAZ.py`: python file that controls preprocessing and downscaling



`input`: the directory containing default input data necessary for running CHAZ. These data are generated during the CHAZ development stage and are used in [Lee et al. 2018](https://doi.org/10.1002/2017MS001186). Check https://drive.google.com/file/d/1v_NyAzqGyfUNolY-FOAg9P0_itExPGEC/view?usp=sharing for input data.
- `bt_*.nc`: basin-wide best track data, created by `bt_gen.py` (active development)
- `bt_global_predictors.nc`: CHAZ predictors for historical events (from `bt_*.nc`) from monthly ERA5 reanalysis data, created by `calCoefficients.py` (active development)
- `coefficient_meanstd.nc`: containing the mean and standard deviations of the predictors for historical events, created by `calCoefficients.py` (active development)
- `landmask.nc`: landmask data
- `result_l.nc`: regression parameters for near land cases, created by 'calCoefficients.py`
- `result_w.nc`: regression parameters for cases over open water, created by 'calCoefficients.py`


`output`: Directory containing CHAZ output, i.e., synthtic storm events 


`pre`: the directory containing data output from preprocessing and input data to CHAZ
- `YYYY*r1i1p1f1.nc`: intensity model predictors (created by TK)
- `A_*YYYYMM.nc`: covariance matrix (created by TK)
- `Cov*YYYY.nc`: monthly-daily wind covariance (created by TK)
- `PI_*YYYY.mat`
- `TCGI_CRH_PI_*YYYY.mat`


`preprocess_ERA5`: directory containing scripts to download ERA5 data and convert 6-hourly data into daily data. 



`src`: Directory containing source code
- `bt_gen`: used to create `bt_*.nc` from IBTrACS data if user does not have best track files (must execute seperately from CHAZ, to be implemented)
- `calCoefficients.py`: Script to create `coefficient_meanstd.nc`, `result_l`, `result_w, `bt_global_predictors.nc` if user does not have these files (must execute separately from CHAZ, to be implemented). Calculates the the mean and standard deviation of the intensity model predictors, global predictors, and regression parameters.
- 

`tools`: Directory containing tools used in source code, CHAZ.py, and Namelist.py




### Downloading data
This iteration of CHAZ uses data from ERA5. To download data in the appropriate format to run CHAZ, run `era5_download.py` to collect monthly information, `era5_download_daily.py` to collect 6-hourly data, and `convert6hourly-daily.py` to convert the 6-hourly data into daily data. This relies on the ECMWF data store's API, which you can learn more about [here](https://confluence.ecmwf.int/spaces/CKB/pages/140380488/How+to+install+and+use+CDS+API+on+macOS).

Once downloaded files, create a `.txt` file so that CHAZ can access this data. 

For example: 

`ls /LOCATION/OF/ERA5/era5data/*.nc > CHAZv2/model-data-fnames.txt`

### Preprocessing

Preprocessing involves populating '\pre' with `r1i1p1_YYYY.nc`, `coefficientmeanstd.nc` and `A_YYYYMM.nc` from `calWindCov.py`, `calA.py`, `getCoefficients.py`, and `preprocess.py`. See [preprocessing example](https://drive.google.com/drive/folders/1Ro1DmHRd0us0Uz9c8jFpeo1BA2gBw3mm?usp=sharing).



### Set up initial conditions
In order to succesfully run CHAZ, you first must ensure all inital conditions and inputs are available.
They are `TCGI_YYYY.mat`, `r1i1p1_YYYY.nc` (from calPreprocess), `A_YYYYMM.nc` (from calA), `coefficent_meanstd.nc `(`getCoefficient.py`), `bt_global_predictors.nc` (`getCoefficient.py`).


### CHAZ downscaling

1. Changing Global Variables in `Namelist.py`

2. Check pointers to inputs, pre-processed data, and output are correct.

3. Select year range to run CHAZ

4. Determine if you would like to overwrite pre-existing output. 

5. To run, type `$ python CHAZ.py`

See [downscaling example](https://drive.google.com/drive/folders/1UvaZ6W4B4oCNCQtSVq0ayA1bF3S2JJLb?usp=sharing) for reference.

##  Output of CHAZ

The output of CHAZ gives a set of netCDF files with names `[model name]_[year]_ens[ensemble number].nc` where "model name" is the model used (also specified in `Namelist.py`), "year" is the year of the simulation, and "ensemble number" is the ensemble number represented with three digits. 

The dimensions of each netCDF file are lifelength, stormID, and ensembleNum (the specific size of the dimensions can be found using the command `$ ncinfo [filename]`), which returns the following:


```
<type 'netCDF4._netCDF4.Dataset'>
root group (NETCDF4 data model, file format HDF5):
    dimensions(sizes): lifelength(125), stormID(146), ensembleNum(40)
    variables(dimensions): float64 lifelength(lifelength), float64 stormID(stormID), 
    float64 ensembleNum(ensembleNum), float64 latitude(lifelength,stormID), 
    float64 Mwspd(ensembleNum,lifelength,stormID), float64 time(lifelength,stormID), 
    float64 longitude(lifelength,stormID), int64 year(stormID)
    groups:
```



## Disclaimer

The output is not bias-corrected. Some bias correction may be necessary to estimate hurricane activity at the regional level.

## License

MIT License

Copyright (c) [2020] [The Columbia HAZard model (CHAZ) team]; The CHAZ team consists of Drs. Suzana J. Camargo, Chia-Ying Lee, Michael K. Tippett, and Adam H. Sobel from Columbia University. 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


## Publications

C.-Y. Lee, M.K. Tippett, A.H. Sobel, and S.J. Camargo, 2018. An environmentally forced tropical cyclone hazard model. Journal of Advances in Modeling Earth Systems, 10, 233-241, doi: 10.1002/2017MS001186.

A.H. Sobel, C.-Y. Lee, S.J. Camargo, K.T. Mandli, K.A. Emanuel, P. Mukhopadhyay, and M. Mahakur, 2019. Tropical cyclone hazard to Mumbai in the recent historical climate. Monthly Weather Review, 147, 2355-2366, doi: 10.1175/MWR-D-18-0419.1.

C.-Y. Lee, S.J. Camargo, A.H. Sobel, and M.K. Tippett, 2020. Statistical-dynamical downscaling projections of tropical cyclone activity in a warming climate: Two diverging genesis scenarios. Journal of Climate, 33, 4815-4834, doi: 10.1175/JCLI-D-19-0452.1.

P. Hassanzadeh, C.-Y. Lee, E. Nabizadeh, S.J. Camargo, D. Ma, and L. Yeung, 2020. Effects of climate change on the movement of future landfalling Texas tropical cyclones, Nature Communications, 11, 3319, doi: 10.1038/s41467/s41467-020-17130-7.