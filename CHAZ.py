#!/usr/bin/env python
import numpy as np
import pandas as pd
import subprocess
import dask.array as da
import os
import src.module_riskModel as mrisk

import src.module_GenBamPred_v2 as GBP_v2
import Namelist as gv

import src.caldetsto_track_lysis_v2 as CHAZ_detsto_v2

import src.calA as calA

import src.calWindCov_v2 as calWindCov_v2
import src.preprocess_v2 as preprocess_v2

#import src.bt_gen as bt_gen
#import src.calCoefficients as calCoefficients

from tools.util import int2str

import pickle

import time


start = time.time()


global fpath
fpath = gv.pre_path


### not implemented
# #######################
# ### Training      #####
# #######################
# if gv.runTraining:
#    if gv.btgen:
#       ## creates bt_*.nc in input/
#       bt_gen.run_btgen()
#    if gv.calCoefficient:
#       ## creates coefficient_meanstd.nc
#       ## and bt_global_predictors.nc in /input
#       ## TODO: implement overwrite protection
#       calCoefficients.run_calCoefficients()

#######################
### Pre-Processes #####
#######################
if gv.runPreprocess: 

   t0 = time.time()
   if not gv.quiet: print('***PREPROCESSING using CHAZ v2 ***')
   if gv.calWind:
      if not gv.quiet: print('calWind')
      #calWindCov.run_windCov()
      calWindCov_v2.run_windCov()
   t1 = time.time()
   if gv.calpreProcess:
      if not gv.quiet: print('calpreProcess')
      ## highest priority
      #preprocess.run_preProcesses()
      preprocess_v2.run_preProcesses()
   t2 = time.time()
   if gv.calA:
      if not gv.quiet: print('calA')
      ## calA is already pretty speedy (lowest priority)
      calA.run_calA()
   t3 = time.time()

   if not gv.quiet: print(f'Preprocess run times: calWind={t1-t0:.2f}  calpreProcess={t2-t1:.2f}  calA={t3-t2:.2f}')


#######################
### Run CHAZ       ####
#######################
if gv.runCHAZ:
   
   print('*** Running CHAZ v2 ***')
   ### get seeding ratio in this experiment
   if gv.TCGIinput != 'random':
      if not gv.quiet: print ('get Seeding ratio for', gv.Model, gv.ENS)
      ipath = gv.pre_path

      ## 
      ratio = GBP_v2.get_seeding_ratio(ipath, 1981, 2005)
      if not gv.quiet: print ('The seeding ratio is',ratio)

   
   ### running genesis,track, predictors, & intensity
   ### output yearly 
   for ichaz in range(gv.CHAZ_ENS_0, gv.CHAZ_ENS):

      for iy in range(gv.Year1, gv.Year2+1):

         output_fname = f"{gv.output_path}{gv.Model}_{iy}_ens{int2str(ichaz,3)}.nc"
         print(output_fname)
         ### if the file does not exist and we do not want to ovewrite
         ### skip running this section
         if not gv.overwrite and os.path.exists(output_fname):
            print(f'{output_fname} already exists, skipping')
            pass

         ### if overwrite is true, regardless of file existence
         ### or if overwrite is false but the file exists
         ### run this section
         else:
            try:
               ### genesis 
               if gv.calGen:
                  if gv.TCGIinput == 'random':
                     print (iy,'random seeding has not yet tested')
                     climInitDate, climInitLon, climInitLat = GBP_v2.randomSeeding(iy,gv.seedN)
                  else:
                     if not gv.quiet: print (iy, 'TCGI')
                     climInitDate, climInitLon, climInitLat = GBP_v2.getSeeding(fpath, iy, ratio)

                  ## debugging
                  if gv.debugging: 
                     print('saving calGen initiation data')
                     np.save(f'{gv.output_path}climInitDate_{iy}', climInitDate)
                     np.save(f'{gv.output_path}climInitLon_{iy}', climInitLon)
                     np.save(f'{gv.output_path}climInitLat_{iy}', climInitLat)
                     print('saved')

               if gv.calBam:
                  if not gv.quiet: print (iy, 'Bam')
                  fst = GBP_v2.getBam(climInitDate, climInitLon, climInitLat, iy, ichaz)
                  
                  try:
                     fst_file = open(f'{gv.output_path}fst_{iy}', 'wb')
                     pickle.dump(fst, fst_file)
                     fst_file.close()
                     if not gv.quiet: print('fst saved to dictionary')

                  except:
                     print('unable to save fst as dictionary')
                  
                  if not gv.quiet: print (iy, 'calculate predictors')
                  GBP_v2.calPredictors(fst,iy,ichaz)
                  del fst
               
               if gv.calInt:
                  if not gv.quiet: print (iy, 'calculate Intensities')
                  CHAZ_detsto_v2.calIntensity(iy, ichaz)
            except Exception as e:
               print(f'Failed for {iy} and realization {ichaz}')
               print(e)

   dir_name = gv.ipath
   t = os.listdir(dir_name)

   for item in t:
      if item.endswith(".pik"):
            os.remove(os.path.join(dir_name, item))  


end = time.time()

print(f"Total runtime of CHAZ is {np.round(end - start,2)} seconds")