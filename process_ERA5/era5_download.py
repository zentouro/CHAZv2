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
import os
import cdsapi
c = cdsapi.Client()

variables1 = ['u_component_of_wind','v_component_of_wind','relative_humidity', 'temperature']
for ivariable in variables1:
	for iyear in range(2002,2010):
		fname = ivariable+'_'+str(iyear)+'.nc'
		if not os.path.exists(fname): 
			c.retrieve(
				'reanalysis-era5-pressure-levels-monthly-means',
				{
				'format': 'netcdf',
				'product_type': 'monthly_averaged_reanalysis',
				'variable': ivariable,
				'pressure_level': [
					'250','500','700','850',
				],
				'year': str(iyear),
				'month': [
					'01', '02', '03',
					'04', '05', '06',
					'07', '08', '09',
					'10', '11', '12',
				],
				'time': '00:00',
				},
				fname)
				#ivariable+'_'+str(iyear)+'.nc')
		else:
			print(fname+' already exists, skipped.')
