def plt_BT(ax,BT, mini, maxi,pp,finfo,u,v):

	print('[figura_mapa] Plotting the map {}'.format(pp))

	plt.sca(ax)

	#Adding features from Cartopy for the contnents
	ax.coastlines(resolution=land_resolution, color='black', linewidth=1)
	ax.add_feature(land_poly)

	#Plotting the variable
	hcv = ax.contourf(lon, lat, BT, levels = np.linspace(mini,maxi,30), cmap = plt.cm.gnuplot, transform=geo, extend='both')

	#Plotting currents
	k = 7
	lon2,lat2 = np.meshgrid(lon,lat)
	print(lon2.shape)
	lons = lon2[::k,::k]
	lats = lat2[::k,::k]
	us = u[::k,::k]
	vs = v[::k,::k]
	vscale = 2

	ve1 = ax.quiver(lons, lats, us, vs,zorder=10000, scale_units='inches', scale=vscale, transform=geo)
	leg = ax.quiverkey(ve1, 0.93, 0.91, 0.5, '0.5 m/s', zorder=1000, labelpos='S', coordinates='axes', labelsep=0.02)

	trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
	ax.text(0.00, 0.2, finfo , zorder=100, size=12, transform=ax.transAxes + trans, verticalalignment='top', color='k')

    #Delimiting figure axes
	ax.set_extent([lon.min(), lon.max(), -8, 8])

	#Making grids and labels
	g = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, 
		color='gray', alpha=0.5, linestyle='--')
	g.top_labels= False
	g.right_labels = False
	if pp != 3:
		g.bottom_labels= False

	return ax, hcv

def plt_BT2(ax,BT, mini, maxi,pp,finfo,u,v):

	print('[figura_mapa] Plotting the map {}'.format(pp))

	plt.sca(ax)

	#Adding features from Cartopy for the contnents
	ax.coastlines(resolution=land_resolution, color='black', linewidth=1)
	ax.add_feature(land_poly)

	#Plotting the variable
	hcv = ax.contourf(lon, lat, BT, levels = np.linspace(mini,maxi,30), cmap = cmo.cm.balance, transform=geo, extend='both')

	#Plotting currents
	k = 7
	lon2,lat2 = np.meshgrid(lon,lat)
	print(lon2.shape)
	lons = lon2[::k,::k]
	lats = lat2[::k,::k]
	us = u[::k,::k]
	vs = v[::k,::k]
	vscale = 0.5

	ve1 = ax.quiver(lons, lats, us, vs,zorder=1000, scale_units='inches', scale=vscale, transform=geo)
	leg = ax.quiverkey(ve1, 0.93, 0.91, 0.25, '0.25 m/s', zorder=1000, labelpos='S', coordinates='axes', labelsep=0.02)

	trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
	ax.text(0.00, 0.2, finfo , zorder=100, size=12, transform=ax.transAxes + trans, verticalalignment='top', color='k')

    #Delimiting figure axes
	ax.set_extent([lon.min(), lon.max(), -8, 8])

	#Making grids and labels
	g = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, 
		color='gray', alpha=0.5, linestyle='--')
	g.top_labels= False
	g.right_labels = False
	if pp != 3:
		g.bottom_labels= False

	return ax, hcv

def UV_mean(years,d):
	for i,y in enumerate(years):
		if i == 0:
			f1 = xr.open_dataset('/data/anamariani/Data/GLORYS/uv_daily_mean/grid0p25/{}.nc'.format(y), chunks = {'time': 30, 'latitude': 5, 'longitude': 10, 'depth':1})
			f1 = f1.sel(time=slice('{}-05-01'.format(y),'{}-10-01'.format(y)), longitude=slice(-60.,15.)).sel(depth=d, method='nearest')
		else:
			f = xr.open_dataset('/data/anamariani/Data/GLORYS/uv_daily_mean/grid0p25/{}.nc'.format(y), chunks = {'time': 30, 'latitude': 5, 'longitude': 10, 'depth':1})
			f = f.sel(time=slice('{}-05-01'.format(y),'{}-10-01'.format(y)), longitude=slice(-60.,15.)).sel(depth=d, method='nearest')
			f1 = xr.concat([f1,f],dim='time')
		print('[UV_mean] Year {} done!'.format(y))
	print('[UV_mean] Calculating the mean')
	u = f1.uo.mean(dim='time').values
	print(u.shape)
	v = f1.vo.mean(dim='time').values
	print(v.shape)
	return u,v

def sst_mean(years):
	f = xr.open_dataset('/data/anamariani/Data/GLORYS/ts_daily_mean/sst_all/sst_GLORYS_0p25.nc')
	f = f.sel(lon=slice(-60.,15.))
	print(f.lon.shape)
	for i,y in enumerate(years):
		if i == 0:
			f1 = f.sel(tempo=slice('{}-05-01'.format(y),'{}-10-01'.format(y)))
		else:
			f2 = f.sel(tempo=slice('{}-05-01'.format(y),'{}-10-01'.format(y)))
			f1 = xr.concat([f1,f2],dim='tempo')
		print('[sst_mean] Year {} done!'.format(y))
	print('[sst_mean] Calculating mean')
	sst = f1.sst.mean(dim='tempo').values
	return sst


import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.transforms as mtransforms
import matplotlib.colors as mcolors
import matplotlib.ticker as tformat
import cmocean as cmo

#Image source details
rc('font', family={'family': 'serif', 'serif': ['Times']})
rc('text', usetex=True)

#Loading dataset
f = xr.open_dataset('/data/anamariani/resultados_mestrado/energy_GLORYS/Barotropic_energy.nc')
f = f.sel(longitude=slice(-60.,15.))
lon = f.longitude.values
lat = f.latitude.values

#plt.ion()

#Propriedades para plot em cartopy
pro = ccrs.PlateCarree(lon.mean())
geo = ccrs.PlateCarree() 
land_resolution = '10m'
land_poly = cfeature.NaturalEarthFeature('physical', 'land', 
	land_resolution, edgecolor='k', facecolor=cfeature.COLORS['land'])

#Close figures
plt.clf()
plt.close('all')

fig, ax = plt.subplots(3,1,sharey=True,sharex=True, figsize=(8,6),
	gridspec_kw={'left': 0.14, 'right': 0.9, 'bottom': 0.04 , 'top':0.95, 'wspace':0.01, 'hspace':-0.2},
	subplot_kw={'projection': pro})

ax1 = ax.ravel()

#years = np.array([2002,2009,2011,2015])
years1 = np.arange(1993,2008,1)
sst1 = sst_mean(years1)
u1,v1 = UV_mean(years1,5.)
ax1[0],cor = plt_BT(ax1[0],sst1, 24.5, 28.5,1,'a)', u1,v1)

#years = np.array([1993,1997,1998])
years2 = np.arange(2008,2021,1)
sst2 = sst_mean(years2)
u2,v2 = UV_mean(years2,5.)
ax1[1],cor = plt_BT(ax1[1],sst2, 24.5, 28.5,2,'b)',u2,v2)

cby = fig.add_axes([0.91,0.36,0.008,0.55])
CB = plt.colorbar(cor,cax=cby,label='SST [°C]', ticks=np.linspace(24.5,28.5,5))

#years = np.array([1994,1995,1996,1999,2000,2001,2003,2004,2005,2006,2007,2008,2010,2012,2013,2014,2016,2017,2018,2019,2020])
#u,v = UV_mean(years,5.)
u = u2 - u1
v = v2 - v1
sst = sst2 - sst1
ax1[2],cor = plt_BT2(ax1[2],sst, -0.5, 0.5,3,'c)',u,v)

cby = fig.add_axes([0.91, 0.085, 0.008, 0.25]) #[posição a leste, base, largura, altura]
CB = plt.colorbar(cor, cax=cby, label='SST bias [°C]', ticks=np.linspace(-0.5,0.5,5))#,extend='both')

fig.savefig('/home/anamariani/Documents/Resultados/figuras_teste_GLORYS/Composite_SST_UV.png', bbox_inches='tight', dpi=300)

plt.show()

