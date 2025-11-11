def get_aux():
	f = xr.open_dataset('/data/anamariani/Data/GLORYS/ts_daily_mean/1993-01.nc', chunks={'time':30, 'depth':1, 'longitude':6, 'latitude':3})
	latitude = f.latitude.values[::3]
	longitude = f.longitude.values[::3]
	depth = f.depth.values
	return latitude,longitude,depth


import numpy as np
import xarray as xr
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import rc

#rc('font', family={'family': 'serif', 'serif': ['Times'],'size': 10})
#rc('text', usetex=True)

# Selecting variables
years = np.arange(1993,2021,1)
lats,lons,depth = get_aux()
BT_series = np.ones((len(years),len(depth),len(lats),len(lons)))*np.nan


for iy,y in enumerate(years):
	print('Calculating barotropic energy for year:{}'.format(y))
	for i_d,d in enumerate(depth):
		#Reading u' and v' from this year at depth 5m:
		fvl = xr.open_dataset('/data/anamariani/resultados_mestrado/energy_GLORYS/v_30_filtered_D{:.2f}.nc'.format(d))
		ful = xr.open_dataset('/data/anamariani/resultados_mestrado/energy_GLORYS/u_30_filtered_D{:.2f}.nc'.format(d))

		#Selecting only the variables from a year from may to september 
		d0 =  dt.date.toordinal(dt.date(y,5,1)) #dt.date.toordinal(dt.date(y,1,1))
		d1 = dt.date.toordinal(dt.date(y,10,1)) #dt.date.toordinal(dt.date(y,12,31))

		vl = fvl.v_R_30.sel(time=slice(d0,d1))
		ul = ful.u_R_30.sel(time=slice(d0,d1))

		ulvl = vl*ul
		ulvl_m = ulvl.mean(dim='time')

		#Calculating mean from original data
		#f1 = xr.open_mfdataset('/data/anamariani/Data/GLORYS/uv_daily_mean/grid0p25/*.nc', chunks = {'time': 30, 'latitude': 81, 'longitude': 100, 'depth':1},parallel=True)
		#u = f1.uo.sel(time=slice('{}-05-01'.format(y),'{}-10-01'.format(y))).sel(depth=d, method='nearest')
		#u_m = u.mean(dim='time')

		#Calculating mean from filtered data
		f1 = xr.open_dataset('/data/anamariani/resultados_mestrado/energy_GLORYS/u_annual_longterm_filtered_D{:.2f}.nc'.format(d))
		u_a = f1.u_annual.sel(time=slice(d0,d1)).mean(dim='time')
		u_long = f1.u_long.sel(time=slice(d0,d1)).mean(dim='time')
		u_m = f1.u_mean + u_a + u_long 


		d2m = 111195 # ° to m
		d2r = np.deg2rad # ° to rad
		lon, lat = np.meshgrid(lons, lats)
		lon_m, lat_m = lon*d2m, lat*d2m
		lon_m = lon_m*np.cos(d2r(lat))
		dy,_ = np.gradient(lat_m, edge_order=2)
		_, dx = np.gradient(lon_m, edge_order=2)

		dudy, _ = np.gradient(u_m.values, edge_order=2)
		dudy = dudy/dy

		BT = -ulvl_m*dudy

		fd = xr.open_dataset('/data/anamariani/resultados_mestrado/energy_GLORYS/rho_mean_D{:.2f}.nc'.format(-d))
		rho_m = fd.rho_mean.values

		BT_series[iy,i_d,:,:] = BT*rho_m
		print('Depth: {:.0f} ready!'.format(d))

print('Salvando dataset:')
ds = xr.Dataset(
		data_vars=dict(
			BT_mean = (['time','depth','latitude', 'longitude'], BT_series),
			),
		coords=dict(latitude=(['latitude'], lats), longitude=(['longitude'], lons), depth=(['depth'], depth), time=(['time'],years))
		)
ds.to_netcdf('/data/anamariani/resultados_mestrado/energy_GLORYS/Barotropic_energy.nc')
print('BT_mean saved')
print('Done!')
