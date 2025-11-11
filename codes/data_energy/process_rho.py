
def TS_to_rho(depth,lat,lon,thetao,so):
	'''
	Calculates density from 
	pratical salinity (so) and
	potential temperature (thetao)
	'''
	#estimate pressure from depth
	p = gsw.conversions.p_from_z(depth, lat, geo_strf_dyn_height=0, sea_surface_geopotential=0)

	#estimate absolute salinity and conservative temperature
	SA = gsw.conversions.SA_from_SP(so, p, lon, lat)
	CT = gsw.conversions.CT_from_pt(SA, thetao)

	#estimate pho from SA and CT
	rho = gsw.density.rho(SA, CT, p)

	return rho

def calculate_mbporchunk(f1):
	chunk_shape = tuple(c[0] for c in f1.thetao.chunks)
	n_elements = np.prod(chunk_shape)
	bits_per_element = np.dtype(f1.thetao.dtype).itemsize * 8
	chunk_size_bits = n_elements * bits_per_element
	chunk_size_megabits = chunk_size_bits / 1e6
	print('{} MB/chunk'.format(chunk_size_megabits))
	return chunk_size_megabits

def get_aux():
	f = xr.open_dataset('/data/anamariani/Data/GLORYS/ts_daily_mean/data_por_lat/lat-0.25_allYearsDepth.nc', chunks={'time':365, 'depth':1, 'longitude':6})
	time = f.time
	longitude = f.longitude.values[::3]
	depth = f.depth.values
	f = xr.open_dataset('/data/anamariani/Data/GLORYS/ts_daily_mean/1993-01.nc', chunks={'time':30, 'depth':1, 'longitude':6, 'latitude':3})
	latitude = f.latitude.values[::3]
	return time,latitude,longitude,depth

def get_zg(rho):
    print('[get_zg] Masking nan values lat={:} ...\n'.format(lat))
    zg = np.ma.masked_invalid(rho)
    zg.set_fill_value(0.0)
    return zg

def mask_basin(lat, zg):
	try:
		f_ssh_alt = xr.open_dataset('/data/anamariani/resultados_mestrado/dados_por_lat_filtrados/semi-auto_new/hovs_filt_atlantic_{}_filtro_semi-auto_1.nc'.format(lat+0.125))
	except FileNotFoundError:
		try: 
			f_ssh_alt = xr.open_dataset('/data/anamariani/resultados_mestrado/dados_por_lat_filtrados/semi-auto_new/hovs_filt_atlantic_{}_filtro_semi-auto_2.nc'.format(lat+0.125))
		except FileNotFoundError:
			f_ssh_alt = xr.open_dataset('/data/anamariani/resultados_mestrado/dados_por_lat_filtrados/semi-auto_new/hovs_filt_atlantic_{}_filtro_semi-auto_3.nc'.format(lat+0.125))
	lon_alt = f_ssh_alt.lon.values
	xio = lons
	nxio = len(xio)
	z = zg[:, :].filled()
	#selecting only longitude values from altimeter filtered data
	i0 = np.where(lons >= lon_alt[0])[0][0]
	i1 = np.where(lons >= lon_alt[-1])[0][0] + 1 
	xio = xio[i0:i1]
	z = z[:, i0:i1]

	return (z, xio,i0,i1)

def plot_hovs(z_c,vx,xio,time):
    '''
    Plots the Hovmollers, variables are specific to this program
    '''
    rc('image', cmap='RdBu_r')
    rc('font', family={'family': 'serif', 'serif': ['Times'],
                  'size': 10})
    rc('text', usetex=True)

    fig, ax = plt.subplots(nrows=1, ncols=len(z_c), num=3, clear=True, sharey=True,
                           figsize=(16, 10))
    ax0 = np.ravel(ax) #(?) Porque unidimensionalizar o eixo? - acho que não precisa disso aqui

    for a, key in enumerate(z_c.keys()):
        zm = z_c[key].mean()
        zs = z_c[key].std()
        if a <= 1:
            vmin = zm - 2*zs
            vmax = zm + 2*zs
        else:
            vmin = -3*zs
            vmax = 3*zs
        pc = ax0[a].pcolormesh(xio, date2num(time), z_c[key],
                               vmin=vmin, vmax=vmax,
                               shading='auto')
        plt.colorbar(pc, ax=ax0[a], orientation='horizontal',
                     extend='both', pad=0.04, fraction=0.03, format='%d',
                     ticks=np.linspace(vmin, vmax, 5))
# format the ticks etc.
        mkey = key.replace('_', ' ')
        ax0[a].set_title(r'{:}, $\sigma^2_p$={:.0f}'.format(
            mkey.capitalize(), vx[key]))
        ax0[a].xaxis.set_major_locator(plt.MultipleLocator(
            int((xio[-1]-xio[0])/20)*10))
        ax0[a].xaxis.set_minor_locator(plt.MultipleLocator(
            int((xio[-1]-xio[0])/20)*5))
        if a == 0:
            iyea = YearLocator()
            years_fmt = DateFormatter('%Y')
        ax0[a].xaxis.set_major_formatter(StrMethodFormatter(r"{x:.0f}$^\circ$W"))
        ax0[a].yaxis.set_major_locator(iyea)
        ax0[a].yaxis.set_major_formatter(years_fmt)
        ax0[a].yaxis.grid(alpha=0.5)
        ax0[a].xaxis.grid(alpha=0.5)
    plt.tight_layout(w_pad=0, h_pad=1)
    return fig, ax0
# -----------------------------------------------------------------------------

import xarray as xr
import numpy as np
import pandas as pd
import datetime as dt
import gsw
import matplotlib.pyplot as plt
from collections import OrderedDict as od
from fir import maxvar, f_longterm, f_annual, varexp
from class_filter import filter_wave
from matplotlib import rc
from matplotlib.dates import YearLocator, DateFormatter, date2num
from matplotlib.ticker import (StrMethodFormatter, MultipleLocator)

time,lats,lons,dep = get_aux()
time_dt = time.values.astype('datetime64[ms]').astype('O')
jtime = [dt.date.toordinal(tt) for tt in time_dt]
jtime = np.asarray(jtime).astype('int32')

d_w = pd.read_csv('/data/anamariani/resultados_mestrado/dados_por_lat_filtrados/dados_filtragem_ondas.csv')
f_lat = d_w['Lat'].values

# minimum number of points in the longitudinal direction worth processing
minxlen = 25

delt = 1     # temporal grid spacing in days
delx = 0.25  # longitudinal grid spacng in degrees

for dd in range(len(dep)):
	print('DEPTH: {:.0f}'.format(dep[dd]))

	rho_ori = np.ones((len(time),len(lats),len(lons)))*np.nan
	rho_mean = np.ones((len(lats),len(lons)))*np.nan
	rho_R_30 = np.ones((len(time),len(lats),len(lons)))*np.nan

	for ila, lat in enumerate(lats):
		print('[main] Reading dataset from lat {:.2f}...'.format(lat))
		f = xr.open_dataset('/data/anamariani/Data/GLORYS/ts_daily_mean/data_por_lat/lat{:.2f}_allYearsDepth.nc'.format(lat), chunks={'time':365, 'depth':1, 'longitude':9})
		f = f.isel(longitude=range(len(f.longitude))[::3], depth=dd)
		_ = calculate_mbporchunk(f)

		depth = -dep[dd]

		print('[main] Processing variables ...')
		#selecting the T and S values
		so = f.so.values
		so = so.squeeze()
		thetao = f.so.values
		thetao = thetao.squeeze()

		lons_2, _ = np.meshgrid(lons,jtime)

		print('[main] Estimate rho ..')
		rho = TS_to_rho(depth,lat,lons_2,thetao,so)
		rho_ori[:,ila,:] = rho
		rho_mean[ila,:] = np.nanmean(rho, axis=0)
		print(np.nanmean(rho_mean[ila,:]))

		if lat > -8. and lat < 8.:
				if lat > f_lat[f_lat>0].min() or lat < f_lat[f_lat<0].max():
					zs = od()
					print('[main] Subtracting mean ...')
					rho = rho - rho_mean[ila,:]
					zg = get_zg(rho)
					(z, xio, xi0,xi1) = mask_basin(lat, zg)
					if np.isnan(z).any():
						print(np.where(np.isnan(z)==True))

					#sinal original
					zs['ori'] = z.copy()

					#sinal longo termo
					zs['longterm'] = f_longterm(z, ny=7, delt=delt)
					zs['longterm'] = maxvar(zs['longterm'], z)*zs['longterm']

					z = zs['ori'] - zs['longterm']
					print('[main] Long term signal extracted.')

					#sinal anual
					zs['annual'] = f_annual(z, delt)
					zs['annual'] = maxvar(zs['annual'], z)*zs['annual']

					z = z - zs['annual']
					print('[main] Annual signal extracted.')

					#sinal R_183
					T = round(183)
					L = -round(d_w[d_w['Lat']==lat+0.125]['L_R_183'].values[0])

					fwave = filter_wave(delx, delt, lat, z, T, L, zs['ori'])
					zs['R_183'] = fwave.matriz_filtro()

					z = z - zs['R_183']
					print('[main] R_183 signal extracted.')

					if lat >= -2.0 and lat < 2.625:
						T = round(d_w[d_w['Lat']==lat+0.125]['P_K_90'].values[0])
						L = round(d_w[d_w['Lat']==lat+0.125]['L_K_90'].values[0])

						fwave = filter_wave(delx, delt, lat, z, T, L, zs['ori'])
						zs['K_90'] = fwave.matriz_filtro()

						z = z - zs['K_90']
						print('[main] K_90 signal extracted.')

					if lat > 0:
						#sinal R_121
						T = round(121)
						L = -round(d_w[d_w['Lat']==lat+0.125]['L_R_121'].values[0])
						fwave = filter_wave(delx, delt, lat, z, T, L, zs['ori'])
						zs['R_121'] = fwave.matriz_filtro()

						z = z - zs['R_121']
						print('[main] R_121 signal extracted.')

					#sinal de 30 a 50 dias:
					f_w30 = np.load('/data/anamariani/resultados_mestrado/dados_por_lat_filtrados/selected_waves_ori/ondas_selecionadas_lat{}.npz'.format(lat+0.125))
					T = round(np.nanmean(f_w30['T']))

					Cp = np.nanmean(f_w30['Cp'].ravel())
					L = round(Cp*T)

					fwave = filter_wave(delx, delt, lat, z, T, L, zs['ori'])
					zs['R_30'] = fwave.matriz_filtro()
					rho_R_30[:,ila,xi0:xi1] = zs['R_30']
					z = z - zs['R_30']
					print('[main] R_30 signal extracted.')

					#Residuo
					zs['res'] = z

					if lat == 5. or lat == -5. or lat == 2.5 or lat == -2.5:
						vx = od()
						for key in zs.keys():
							vx[key] = 100*varexp(zs['ori'], zs[key])

						fig,ax = plot_hovs(zs,vx,xio,time)
						print('Salvando a figura...')
						fig.savefig('/home/anamariani/Documents/Resultados/figuras_teste_GLORYS/teste_hovs_latitude{:.2f}_filtro_glo_pho_D{:.0f}.png'.format(lat,depth), dpi=300)
						print('Salvo <3')

	print('Salvando dataset:')
	ds = xr.Dataset(
			data_vars=dict(rho_ori=(['time', 'latitude', 'longitude'], rho_ori),
				),
			coords=dict(latitude=(['latitude'], lats), longitude=(['longitude'], lons), time=(['time'], jtime))
			)
	ds.to_netcdf('/data/anamariani/resultados_mestrado/energy_GLORYS/rho_ori_filtered_D{:.2f}.nc'.format(depth))
	print('rho_ori saved')
	ds = xr.Dataset(
			data_vars=dict(rho_R_30= (['time', 'latitude', 'longitude'], rho_R_30),
				),
			coords=dict(latitude=(['latitude'], lats), longitude=(['longitude'], lons), time=(['time'], jtime))
			)
	ds.to_netcdf('/data/anamariani/resultados_mestrado/energy_GLORYS/rho_30_filtered_D{:.2f}.nc'.format(depth))
	print('rho_30 saved')
	ds = xr.Dataset(
			data_vars=dict(
				rho_mean = (['latitude', 'longitude'], rho_mean),
				),
			coords=dict(latitude=(['latitude'], lats), longitude=(['longitude'], lons))
			)
	ds.to_netcdf('/data/anamariani/resultados_mestrado/energy_GLORYS/rho_mean_D{:.2f}.nc'.format(depth))
	print('rho_mean saved')
	print('Done!')




	


