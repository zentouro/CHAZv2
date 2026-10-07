#!/usr/bin/env python
"""
download ear5
4. Under your home folder create a file named .cdsapirc
Here you will need your ERA account.
See details at:
https://cds.climate.copernicus.eu/api-how-to

5. here is an example how the cds generate the script for you
https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels-monthly-means?tab=form
"""
#from pygplib3 import util
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from tools import util

import cdsapi
c = cdsapi.Client()
variables1 = ['v_component_of_wind','u_component_of_wind']
for ivariable in variables1[:]:
	for iyear in range(2000, 2010):
	#for iyear in range(1979,2015):
		for imonth in range(1,13):
			fname = ivariable+'_'+str(iyear)+util.int2str(imonth,2)
			#fname = ivariable+'_'+str(iyear)+util.int2str(imonth,2)+'_6hly.nc'
			### update to check that the _6hrly.nc and _dailymean.nc don't exist
			if not os.path.exists(fname+'_6hly.nc') and not os.path.exists(fname+'_dailymean.nc'):
				c.retrieve(
					'reanalysis-era5-pressure-levels',
					{
					'product_type': 'reanalysis',
					'format': 'netcdf',
					'variable': ivariable,
					'pressure_level': [
						'250', '850',
					],
					'year': str(iyear),
					'month': util.int2str(imonth,2),
					'day': [
						'01', '02', '03',
						'04', '05', '06',
						'07', '08', '09',
						'10', '11', '12',
						'13', '14', '15',
						'16', '17', '18',
						'19', '20', '21',
						'22', '23', '24',
						'25', '26', '27',
						'28', '29', '30',
						'31',
					],
					'time': [
						'00:00', '06:00', '12:00',
						'18:00',
					],
					},
					fname+'_6hly.nc')
			    	#ivariable+'_'+str(iyear)+util.int2str(imonth,2)+'_6hly.nc')
			else:
				print(fname+' already exists, skipped.')


