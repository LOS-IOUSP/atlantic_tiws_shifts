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


def mask_basin(lat, zg):
	try:
		f_ssh_alt = xr.open_dataset('/data/anamariani/resultados_mestrado/dados_por_lat_filtrados/semi-auto_new/hovs_filt_atlantic_{}_filtro_semi-auto_1.nc'.format(lat+0.125))
	except FileNotFoundError:
		try: 
			f_ssh_alt = xr.open_dataset('/data/anamariani/resultados_mestrado/dados_por_lat_filtrados/semi-auto_new/hovs_filt_atlantic_{}_filtro_semi-auto_2.nc'.format(lat+0.125))
		except FileNotFoundError:
			f_ssh_alt = xr.open_dataset('/data/anamariani/resultados_mestrado/dados_por_lat_filtrados/semi-auto_new/hovs_filt_atlantic_{}_filtro_semi-auto_3.nc'.format(lat+0.125))
	lon_alt = f_ssh_alt.lon.values
	xio = x
	nxio = len(xio)
	z = zg[:, :].filled()
	#selecting only longitude values from altimeter filtered data
	i0 = np.where(x >= lon_alt[0])[0][0]
	i1 = np.where(x >= lon_alt[-1])[0][0] + 1 
	xio = xio[i0:i1]
	z = z[:, i0:i1]

	return (z, xio,i0,i1)

# -----------------------------------------------------------------------------

def get_zg(ila,uo):
    print('[get_zg] Selecting latitude {:} ...\n'.format(lat))
    # convert from m to mm
    zg = uo[:,ila,:]
    zg = np.ma.masked_invalid(zg)
    zg.set_fill_value(0.0)
    return zg
#------------------------------------------------------------------------------


import xarray as xr
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from collections import OrderedDict as od
from fir import maxvar, f_longterm, f_annual, varexp
from class_filter import filter_wave
from matplotlib import rc
from matplotlib.dates import YearLocator, DateFormatter, date2num
from matplotlib.ticker import (StrMethodFormatter, MultipleLocator)

f1 = xr.open_mfdataset('/data/anamariani/Data/GLORYS/uv_daily_mean/grid0p25/*.nc', chunks = {'time': 183, 'latitude': 10, 'longitude': 100, 'depth':1}, parallel=True)
depths = f1.depth.values

for i_d, depth in enumerate(depths):
	f = f1.isel(depth=i_d)
	print('Depth:{:.0f} - {}/{}'.format(depth,i_d+1,len(depths)-1))
	mean_t = f.uo.mean(dim='time')
	uo = f.uo - mean_t
	uo = uo.values
	print(mean_t)
	lats = f.latitude.values
	x = f.longitude.values
	nx = x.shape[0]
	time = f.time.values
	time = time.astype('datetime64[ms]').astype('O')
	nt = len(time)
	jtime = [dt.date.toordinal(time[j]) for j in range(nt)] #jtime = data na forma de numeros inteiros
	jtime = np.asarray(jtime).astype('int32')

	d_w = pd.read_csv('/data/anamariani/resultados_mestrado/dados_por_lat_filtrados/dados_filtragem_ondas.csv')
	f_lat = d_w['Lat'].values

	# minimum number of points in the longitudinal direction worth processing
	minxlen = 25

	delt = 1     # temporal grid spacing in days
	delx = 0.25  # longitudinal grid spacng in degrees

	#mapas para cada componente:
	m_ori = np.ones(f.uo.shape)*np.nan
	m_long = np.ones(f.uo.shape)*np.nan
	m_annual = np.ones(f.uo.shape)*np.nan
	#m_R_183 =  np.ones(f.uo.shape)*np.nan
	#m_R_121 = np.ones(f.uo.shape)*np.nan
	m_R_30 = np.ones(f.uo.shape)*np.nan
	#m_res = np.ones(f.uo.shape)*np.nan


	for ila, lat in enumerate(lats):
		if lat > -8. and lat < 8.:
			if lat > f_lat[f_lat>0].min() or lat < f_lat[f_lat<0].max():
				zs = od()
				print('[main] Processing lat={:}'.format(lat))
				zg = get_zg(ila, uo)
				(z, xio, xi0,xi1) = mask_basin(lat, zg)
				if np.isnan(z).any():
					print(np.where(np.isnan(z)==True))

				#sinal original
				zs['ori'] = z.copy()
				m_ori[:,ila,xi0:xi1] = zs['ori']

				#sinal longo termo
				zs['longterm'] = f_longterm(z, ny=7, delt=delt)
				zs['longterm'] = maxvar(zs['longterm'], z)*zs['longterm']
				m_long[:,ila,xi0:xi1] = zs['longterm']

				z = zs['ori'] - zs['longterm']
				print('[main] Long term signal extracted.')

				#sinal anual
				zs['annual'] = f_annual(z, delt)
				zs['annual'] = maxvar(zs['annual'], z)*zs['annual']
				m_annual[:,ila,xi0:xi1] = zs['annual']

				z = z - zs['annual']
				print('[main] Annual signal extracted.')

				#sinal R_183
				T = round(183)
				L = -round(d_w[d_w['Lat']==lat+0.125]['L_R_183'].values[0])

				fwave = filter_wave(delx, delt, lat, z, T, L, zs['ori'])
				zs['R_183'] = fwave.matriz_filtro()
				#m_R_183[:,ila,xi0:xi1] = zs['R_183']

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
					#m_R_121[:,ila,xi0:xi1] = zs['R_121']

					z = z - zs['R_121']
					print('[main] R_121 signal extracted.')

				#sinal de 30 a 50 dias:
				f_w30 = np.load('/data/anamariani/resultados_mestrado/dados_por_lat_filtrados/selected_waves_ori/ondas_selecionadas_lat{}.npz'.format(lat+0.125))
				T = round(np.nanmean(f_w30['T']))

				Cp = np.nanmean(f_w30['Cp'].ravel())
				L = round(Cp*T)

				fwave = filter_wave(delx, delt, lat, z, T, L, zs['ori'])
				zs['R_30'] = fwave.matriz_filtro()
				m_R_30[:,ila,xi0:xi1] = zs['R_30']
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
					fig.savefig('/home/anamariani/Documents/Resultados/figuras_teste_GLORYS/teste_hovs_latitude{:.2f}_filtro_glo_u_D{:.0f}.png'.format(lat,depth), dpi=300)
					print('Salvo <3')
				

	m_ori = np.ma.masked_where(np.isnan(f.uo),m_ori)
	m_ori.set_fill_value(np.nan)
	m_long = np.ma.masked_where(np.isnan(f.uo),m_long)
	m_long.set_fill_value(np.nan)
	m_annual = np.ma.masked_where(np.isnan(f.uo),m_annual)
	m_annual.set_fill_value(np.nan)
	#m_R_183 = np.ma.masked_where(np.isnan(f.uo),m_R_183)
	#m_R_183.set_fill_value(np.nan)
	#m_R_121 = np.ma.masked_where(np.isnan(f.uo),m_R_121)
	#m_R_121.set_fill_value(np.nan)
	m_R_30 = np.ma.masked_where(np.isnan(f.uo),m_R_30)
	m_R_30.set_fill_value(np.nan)
	#m_res = np.ma.masked_where(np.isnan(f.uo),m_res)
	#m_res.set_fill_value(np.nan)

	print('Salvando dataset:')
	ds = xr.Dataset(
			data_vars=dict(u_ori=(['time', 'latitude', 'longitude'], m_ori),
				),
			coords=dict(latitude=(['latitude'], lats), longitude=(['longitude'], x), time=(['time'], jtime))
			)
	ds.to_netcdf('/data/anamariani/resultados_mestrado/energy_GLORYS/u_ori_filtered_D{:.2f}.nc'.format(depth))
	print('u_ori saved')
	ds = xr.Dataset(
			data_vars=dict(u_R_30= (['time', 'latitude', 'longitude'], m_R_30),
				),
			coords=dict(latitude=(['latitude'], lats), longitude=(['longitude'], x), time=(['time'], jtime))
			)
	ds.to_netcdf('/data/anamariani/resultados_mestrado/energy_GLORYS/u_30_filtered_D{:.2f}.nc'.format(depth))
	print('u_30 saved')
	ds = xr.Dataset(
			data_vars=dict(u_long= (['time', 'latitude', 'longitude'], m_long),
				u_annual= (['time', 'latitude', 'longitude'], m_annual),
				u_mean = (['latitude', 'longitude'], mean_t.values),
				),
			coords=dict(latitude=(['latitude'], lats), longitude=(['longitude'], x), time=(['time'], jtime))
			)
	ds.to_netcdf('/data/anamariani/resultados_mestrado/energy_GLORYS/u_annual_longterm_filtered_D{:.2f}.nc'.format(depth))
	print('u_long and annual saved')
	print('Done!')

	


