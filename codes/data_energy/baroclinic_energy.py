def get_aux():
	f = xr.open_dataset('/data/anamariani/Data/GLORYS/ts_daily_mean/1993-01.nc', chunks={'time':30, 'depth':1, 'longitude':6, 'latitude':3})
	latitude = f.latitude.values[::3]
	longitude = f.longitude.values[::3]
	depth = f.depth.values
	return latitude,longitude,depth

def calc_divergence(u, v, lat, lon):
    """
    Calculates the divergent of a bidimensional vector from a gridded data.
    Variables:
        - u: bidimensional marix with the zonal components (2D)
        - v: bidimensional marix with the meridional components (2D)
        - lat: latitude matrix (2D)
        - lon: longitude matrix (2D)
    Return:
        - Bidimensional matrix with the divergent values
    """
    dx = np.gradient(lon, axis=1)*111195*np.cos(lat*np.pi/180)
    dy = np.gradient(lat, axis=0)*111195
    dudx = np.gradient(u, axis=1)/dx
    dvdy = np.gradient(v, axis=0)/dy
    return dudx + dvdy

def calc_w(div,i_d):
	dz = (-depth[i_d-1]) - (-depth[i_d+1])
	w = div*dz
	return w


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
lon2, lat2 = np.meshgrid(lons,lats)
BC_series = np.ones((len(years),len(depth),len(lats),len(lons)))*np.nan
g = 9.81

for iy,y in enumerate(years):
	print('Calculating baroclinic energy for year:{}'.format(y))
	for i_d,d in enumerate(depth):
		if (i_d%2 == 1) and (i_d < len(depth)-1):
			#print(i_d)
			fvl = xr.open_dataset('/data/anamariani/resultados_mestrado/energy_GLORYS/v_30_filtered_D{:.2f}.nc'.format(d))
			ful = xr.open_dataset('/data/anamariani/resultados_mestrado/energy_GLORYS/u_30_filtered_D{:.2f}.nc'.format(d))

			#Selecting only the variables from a year from may to september 
			d0 =  dt.date.toordinal(dt.date(y,5,1)) #dt.date.toordinal(dt.date(y,1,1))
			d1 = dt.date.toordinal(dt.date(y,10,1)) #dt.date.toordinal(dt.date(y,12,31))

			vl = fvl.v_R_30.sel(time=slice(d0,d1))
			ul = ful.u_R_30.sel(time=slice(d0,d1))

			wl = np.ones(vl.shape)*np.nan

			for tt,time in enumerate(vl.time):
				div = calc_divergence(ul.sel(time=time), vl.sel(time=time), lat2, lon2)
				wl[tt,:,:] = calc_w(div,i_d)

			fp = xr.open_dataset('/data/anamariani/resultados_mestrado/energy_GLORYS/rho_30_filtered_D{:.2f}.nc'.format(-d))
			phol = fp.rho_R_30.sel(time=slice(d0,d1))

			if i_d == 1:
				wl_sum = wl
			else:
				wl_sum = wl_sum+wl

			rholwl_m = wl_sum*phol
			rholwl_m = rholwl_m.mean(dim='time')

			BC_series[iy,i_d,:,:] = -rholwl_m*g
			print('Depth: {:.0f} ready!'.format(d))


print('Salvando dataset:')
ds = xr.Dataset(
		data_vars=dict(
			BC_mean = (['time','depth','latitude', 'longitude'], BC_series),
			),
		coords=dict(latitude=(['latitude'], lats), longitude=(['longitude'], lons), depth=(['depth'], depth), time=(['time'],years))
		)
ds.to_netcdf('/data/anamariani/resultados_mestrado/energy_GLORYS/Baroclinic_energy_sum.nc')
print('BC_mean saved')
print('Done!')











