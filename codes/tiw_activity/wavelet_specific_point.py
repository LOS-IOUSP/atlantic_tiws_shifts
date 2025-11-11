#!/home/ana/anaconda3/bin/python3
# *- coding: utf-8 -*-
"""
Created on Tue Nov 14 14:09 2023

@author: anamariani - adaptado de @polito

Programa que faz um escalograma de um determinado ponto em uma latitude e longitude do espectro filtrado para ondas de 20 a 50 dias do nível do mar 
"""

def ana_cwt(znd,dt,s0,dj,J):
	"""
	* Wavelet transform yields a localized estimate for the amplitude and phase
	of each spectral component in the dataset.
	* Good when the amplitude and phase o fthe harmonic may change in space or time.
	* Tracks the evolution of the signal characteristics through the dataset.
	* Code created by Prof. Dr. Paulo Polito and adapted by
										Arian Dialectaquiz
	
	*Input:
		dt: data frequency of sample 
		s0: initial parameter of displacement
		dj: dj sub-octaves per octave for the domain
		J: J power of two with dj sub-octaves

	"""

	#mother wavelet
	mother = wavelet.Morlet(6)
	Cdelta = mother.cdelta
	alpha = (ac_lag(znd, 2) + ac_lag(znd, 3))/2 #parametro associado a
	#alpha, _, _ = wavelet.ar1(znd)  # Lag-1 autocorrelation for red noise
	N = znd.size
	std = znd.std()  # Standard deviation
	var = std ** 2  # Variance
	print(var)

	np.int = np.int_ #desatualizado no pacote
	wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(znd, dt, dj, s0, J, mother)

	periods = 1 / freqs
	#rectify the power spectrum according 
	#to the suggestions proposed by Liu et al. (2007)
	power = (np.abs(wave)) ** 2
	power /= scales[:, None]
	power /= power.sum()

	cc, pp = np.meshgrid(coi, periods)
	mpower = ma.masked_where(pp > cc, power)

	#ipower = np.mean(mpower.data, axis=0)
	ipower = np.sum(mpower.data, axis=0)
	#ipower = ipower/ipower.max()

	sel = find((periods >= 12) & (periods < 183))
	Cdelta = mother.cdelta
	scale_avg = (scales * np.ones((N, 1))).transpose()
	scale_avg = power / scale_avg  # As in Torrence and Compo (1998) equation 24
	scale_avg = var * dj * dt / Cdelta * scale_avg[sel, :].sum(axis=0)
	scale_avg_signif, tmp = wavelet.significance(znd, dt, scales, sigma_test=2, alpha=alpha, significance_level=0.95, dof=[scales[sel[0]], scales[sel[-1]]], wavelet=mother)

	signif, fft_theor = wavelet.significance(var, dt, scales, alpha=alpha, sigma_test=0,significance_level=0.95, wavelet=mother)#, dof=-1)

	sig95 = np.ones([1, N]) * signif[:, None]

	sig95 = power / sig95

	#print("SIGNIF: {}".format(sig95))
	#print('Significativo:')
	#print(np.where(sig95>1))

	return periods, power, ipower, coi, fft, fftfreqs, sig95, signif, scale_avg, scale_avg_signif

def ac_lag(series, lag):
    # Calcula a média da série
    mean = np.nanmean(series)
    
    # Calcula o desvio padrão da série
    std_dev = np.nanstd(series)
    
    # Calcula a autocorrelação para o lag especificado
    n = len(series)
    acf = np.sum((series[:-lag] - mean) * (series[lag:] - mean)) / ((n - lag) * std_dev ** 2)
    
    return acf

# import sys
import numpy as np
import numpy.ma as ma
from scipy.signal import detrend
import pycwt as wavelet
from scipy.signal import find_peaks
from scipy.stats import chi2
import xarray as xr
import glob
import datetime as da
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LogNorm
import matplotlib.transforms as mtransforms
from matplotlib import rc
from matplotlib import ticker
from pycwt.helpers import find
from statistics import median
import pymannkendall as mk


#Defining initial Variables

#Selecting a specific latitude and longitude
lats = [ 4.375, 4.375,7.375, 7.375, -6.625, -1.875]#[ 4.375, 7.375, 7.375, -6.875]#[7.375, 7.375, 4.375, 4.375]
lons = [-30.875, -23.125,-45.625, -50.125, -16.125, -18.125] #[-30.875, -45.625, -50.125, -18.125]#[-42.625, -50.125, -23.125,-40.125]
#Selecting the filtered data
#componente de 20 a 55 dias
x = np.arange(20,55)
c30 = np.array(['z_R_{}'.format(i) for i in x])

# counter for managing axes and figures
k = 0

for la, lat in enumerate(lats):
	print('Lat: {} Lon:{}'.format(lat,lons[la]))
	#Reading the file of filtered data in that latitude
	name = glob.glob('/data/anamariani/resultados_mestrado/dados_por_lat_filtrados/semi-auto_new/hovs_filt_atlantic_{}_filtro_semi-auto_*.nc'.format(lat))
	f = xr.open_dataset(name[0])

	#Selecting the longitude
	f = f.sel(lon=lons[la])

	#time
	jtime = f.time.values
	nt = len(jtime)
	time = [da.date.fromordinal(jtime[j]) for j in range(nt)]
	td = jtime - jtime[0]

	#Selecting the component of the filtered TIW
	keys = np.array(list(f.keys()))
	comp = np.intersect1d(keys, c30)
	z = f[comp[0]].values
	#z = f['z_ori'].values
	variancia = (np.std(z))**2
	print('Variancia: {}'.format(variancia))

	print(z)
	
	#----------------- wavelet params -------------------------------
	# temporal resolution in days
	dt = 1.0
	s0 = 5.71 #5.71 #11.048738 #3 Starting scale, in this case ~10 days, fine
	# tuned to produce one harmonic near 365.2425 days
	dj = 1.0 / 8  # sub-octaves per octave # 1.0/8
	J = 6.0 / dj  # 6 powers of two with dj sub-octaves #6.0/dj

	# normalize
	z_std = z.std()
	z_mean = z.mean()
	zn = (z-z_mean)/z_std
	znd = detrend(zn)
	#znd = detrend(z)
	#znd = np.random.randn(len(znd))

	# test with constant annual (or bi or semi) signal
	istest = False
	if istest:
		peri = 365
		tt = dt*np.arange(0, len(time))
		om = 2*np.pi/peri
		ztest = 10*np.sin(tt*om)
		znd = znd + ztest

	periods, power, ipower, coi, fft, fftfreqs, sig95, signif, scale_avg, scale_avg_signif = ana_cwt(znd,dt,s0,dj,J)

	#     # =============================================================================
	#     # --- Plot time series
	#     # =============================================================================
	rc('font', family={'family': 'serif', 'serif': ['Times'], 'size': 12})
	rc('text', usetex=True)
	cmap = plt.cm.nipy_spectral#plt.cm.viridis
	#plt.ion()

	figprops = {'num':k, 'figsize':(7, 5), 'clear':True}
	#figprops = {'num':k, 'figsize':(7.2, 4.8), 'clear':True}
	gridspec_kw = {'height_ratios':[2, 4, 2],
	'width_ratios':[5, 1], 'hspace':0.04, 'wspace':0.02,
	'left':0.1, 'right':0.985, 'top':0.88, 'bottom':0.06 }
	fig, ax = plt.subplots(3, 2, **figprops, gridspec_kw=gridspec_kw)
	ax = ax.ravel()
	[a.set_facecolor('#ffffed') for a in ax]
	trans = mtransforms.ScaledTranslation(10/72,-5/72, fig.dpi_scale_trans)

	# ----------------------------------------------------------------------------------------
	# Time series
	if lat > 0:
		tits = r'Wavelet power ($P(t,\omega)$) of $\eta(t)$ at ' + '{:.1f}°N {:.1f}°W'.format(lat,-lons[la])
	else:
		tits = r'Wavelet power ($P(t,\omega)$) of $\eta(t)$ at ' + '{:.1f}°S {:.1f}°W'.format(-lat,-lons[la])
	#fig.suptitle(tits, fontsize=16)
	ax[0].plot(time, znd, linewidth=0.75, color=cmap(0))
	year = mdates.YearLocator()   # every year
	years = mdates.YearLocator(5)
	yearsFmt = mdates.DateFormatter('%Y')
	#months = mdates.MonthLocator(bymonth=(1, 4, 7, 10))   # every month
	#monthsFmt = mdates.DateFormatter('%m')
	ax[0].set_xlim([time[0], time[-1]])
	ax[0].grid('on', which='both')
	ax[0].xaxis.set_major_locator(years)
	ax[0].xaxis.set_major_formatter(yearsFmt)
	ax[0].xaxis.set_minor_locator(year)
	#ax[0].xaxis.set_minor_formatter(yearsFmt)
	ax[0].tick_params(which='both', top='on', labeltop=True, labelbottom=False, grid_color='#343a40',
	grid_alpha=0.5)
	ax[0].tick_params(axis='x', which='minor', labelsize=8, pad=0, grid_linewidth=0.5, grid_alpha=0.5,
	grid_linestyle=':')
	ax[0].tick_params(axis='x', which='major', labelsize=14, pad=4, grid_linewidth=0.8, grid_alpha=0.5,
	grid_linestyle='-')
	#ax[0].set_xlabel(r'Time')
	ax[0].set_ylabel(r'$\eta$ [mm]', labelpad=-4)
	ax[0].text(-0.02,0.28,'a)', zorder=100, size=12.0, transform=ax[0].transAxes + trans, verticalalignment='top',)

	# --------------------------------------------------------------------------------
	# Wavelet Spectrum
	levels = power.max()*np.logspace(-2, 0, 12)
	levels = levels*variancia*100
	#     td = ts.data
	ctws = ax[2].contourf(time, periods, power*variancia*100,levels, norm=LogNorm(), #np.log10(power), np.log10(levels),
					extend='both', cmap=cmap)
	extent = [time[0], time[-1], 0, max(periods)]
	ax[2].contour(time, periods, sig95, [-99, 1], colors='k', linewidths=2,
           extent=extent)
	d1 = da.timedelta(days=1.0)
	ax[2].fill(np.concatenate([time, time[-1:], time[-1:], time[:1], time[:1]]),
			np.concatenate([coi, [1e-9], periods[-1:], periods[-1:], [1e-9]]),
			'#ced4da', alpha=0.2, hatch='x', edgecolor='w')

	ax[2].set_ylabel('Period (days)')
	#Yticks = np.round(365.2425*np.array([1/14, 1/12, 1/10, 1/8, 1/6, 1/4, 1/2])).astype(int)
	Yticks = np.round(np.array([6, 12, 23, 46, 91, 183]))
	ax[2].set_yscale('log')
	ax[2].set_ylim([periods[0], periods[-1]])
	ax[2].set_yticks([])
	ax[2].set_yticks([],minor=True)
	ax[2].yaxis.set_major_locator(ticker.FixedLocator(Yticks))
	ax[2].yaxis.set_major_formatter(ticker.FixedFormatter(Yticks.astype(str)))

	ax[2].xaxis.set_major_locator(years)
	ax[2].xaxis.set_major_formatter(yearsFmt)
	ax[2].xaxis.set_minor_locator(year)
	#ax[2].xaxis.set_minor_formatter(yearsFmt)
	ax[2].tick_params(which='both', top='on', labeltop=False, labelbottom=False, grid_color='white',
						grid_alpha=0.3)
	ax[2].tick_params(axis='x', which='minor', labelsize=8, pad=0, grid_linewidth=0.5,
						grid_linestyle=':')
	ax[2].tick_params(axis='x', which='major', labelsize=14, pad=4, grid_linewidth=0.8,
						grid_linestyle='-')

	ax[2].tick_params(axis='y', which='minor', left='off')
	#ax[2].tick_params(axis='x', labelbottom=False)
	ax[2].grid(visible=True, axis='both', which='both')
	ax[2].text(-0.02,0.16,'b)', zorder=100, size=12.0, transform=ax[2].transAxes + trans, verticalalignment='top',color='w')

	# --------------------------------------------------------------------------------
	# FFT
	pw = np.abs(fft)**2
	per = 1/fftfreqs
	i = np.where(per>=periods[-1])[0][-1]
	j = np.where(per<=periods[0])[0][0]
	pw = pw[i:j+1]
	per=per[i:j+1]

	ax[3].loglog(pw, per, linewidth=0.75, color=cmap(0.2))
	ax[3].set_xlim([pw.min(), pw.max()])
	ax[3].set_ylim([periods[0], periods[-1]])
	ax[3].yaxis.tick_right()
	ax[3].set_yticks([])
	ax[3].set_yticks([],minor=True)
	ax[3].grid(axis='y', which='major', color='black', alpha=0.2)
	ax[3].grid(axis='x', which='major', color='black', alpha=0.2)
	ax[3].yaxis.set_major_locator(ticker.FixedLocator(Yticks))
	ax[3].set_yticklabels([])

	#ax[3].yaxis.set_major_formatter(ticker.FixedFormatter(Yticks.astype(str)))

	Xticks = np.array([0.01, 0.1, 1, 10, 100, 1000])
	Sxticks = [r'$10^{'+f'{np.log10(x):.0f}'+r'}$' for x in Xticks]
	ax[3].xaxis.set_major_locator(ticker.FixedLocator(Xticks))
	ax[3].xaxis.set_major_formatter(ticker.FixedFormatter(Sxticks))
	ax[3].tick_params(axis='y', which='major', right='on', left='off')
	ax[3].tick_params(axis='x', which='major', bottom='on', rotation=45, labelsize=10)
	ax[3].set_title(r'PSD')
	ax[3].text(-0.07,1.0,'d)', zorder=100, size=12.0, transform=ax[3].transAxes + trans, verticalalignment='top')

	CB = plt.colorbar(ctws,ax=ax[3], label=r'$|W(s)|^2$ [mm²/dia]', format='%.2f',fraction =0.15, aspect = 40)
	CB.ax.set_title(r'$\times {}$'.format('10^{-2}'), fontsize=13,  loc='left')
	CB.set_ticks([levels[0],levels[3],levels[6],levels[9],levels[-1]])

	# --------------------------------------------------------------------------------
	# Mean wavelet powerb
	ax[4].plot(time, ipower*variancia,linewidth=0.75, color=cmap(0.1))

	print('Integral ipower: {}'.format(np.sum(ipower*variancia)))

	p,_ = find_peaks(ipower)
	mediana = median(ipower)
	print('Mediana: {:.2e}'.format(mediana))
	p5 = [ip for ip in p if ipower[ip] > mediana]
	peaks = ipower[p5]
	time = np.array(time)
	tpeaks = time[p5]

	mka = mk.yue_wang_modification_test(ipower)
	slope = mka.slope

	zfit = slope*variancia*td + mka.intercept
	print('Tendencia: {}'.format(slope*365*10))

	ax[4].scatter(tpeaks, peaks*variancia, s=1.5, c='k')
	#ax[4].plot(time,zfit, ls=':', c='r')
	#ax[4].plot(time, scale_avg*10000,linewidth=0.75, color=cmap(0.1))
	#ax[4].axhline(scale_avg_signif*10000, color='k', linestyle='--', linewidth=1.)

	ax[4].set_xlim([time[0], time[-1]])
	ax[4].grid('on', which='both')
	ax[4].xaxis.set_major_locator(years)
	ax[4].xaxis.set_major_formatter(yearsFmt)
	ax[4].xaxis.set_minor_locator(year)
	#ax[4].axhline(y=0.6, c='r', lw=1.2)
	#ax[4].xaxis.set_minor_formatter(yearsFmt)
	ax[4].tick_params(which='both', top='on', labeltop=False, labelbottom=True, grid_color='#343a40',
					grid_alpha=0.5)
	ax[4].tick_params(axis='x', which='minor', labelsize=8, pad=2, grid_linewidth=0.5, grid_alpha=0.5,
					grid_linestyle=':')
	ax[4].tick_params(axis='x', which='major', labelsize=14, pad=5, grid_linewidth=0.8, grid_alpha=0.5,
					grid_linestyle='-')
	ax[4].set_ylabel(r'$\int |W|^2\,d\omega$', labelpad=1) #$\times 10^{-4}$', labelpad=1) #$\times 10^{-4}$
	ax[4].text(-0.02,0.98,'c)', zorder=100, size=12.0, transform=ax[4].transAxes + trans, verticalalignment='top')


	fig.delaxes(ax[1])
	fig.delaxes(ax[5])
	plt.savefig('/home/anamariani/Documents/Resultados/figuras_projeto/capitulos/cap3/wavelet_tiw_alt_lat{}_lon{}_espectro_original.png'.format(lat, lons[la]), dpi=300, bbox_inches='tight')
	#plt.savefig('/home/anamariani/Documents/Resultados/figuras_teste/wavelets/wavelet_tiw_alt_lat{}_lon{}_3.png'.format(lat, lons[la]), dpi=300)


	fig2 = plt.figure()
	fig,ax = plt.subplots(figsize=(6, 6))
	CB = plt.colorbar(ctws,ax=ax, label=r'$|W(s)|^2$ [mm²/dia]', format='%.2f',fraction =0.015, aspect = 40)
	CB.ax.set_title(r'$\times {}$'.format('10^{-2}'), fontsize=13,  loc='left')
	CB.set_ticks([levels[0],levels[3],levels[6],levels[9],levels[-1]])
	#CB.set_ticks([np.log10(levels[0]),np.log10(levels[3]),np.log10(levels[6]),np.log10(levels[9]),np.log10(levels[-1])])
	#CB.ax.set_yticklabels([r'$10^{:.1f}$'.format(np.log10(levels[0])),r'$10^{:.1f}$'.format(np.log10(levels[3])),
	#	r'$10^{:.1f}$'.format(np.log10(levels[6])),r'$10^{:.1f}$'.format(np.log10(levels[9])),r'$10^{:.1f}$'.format(np.log10(levels[-1]))])

	ax.remove()
	#plt.savefig('/home/anamariani/Documents/Resultados/figuras_projeto/capitulos/cap3/wavelet_tiw_alt_lat{}_lon{}_plot_onlycbar_espectro_original.png'.format(lat, lons[la]),dpi=300,bbox_inches='tight')
	#plt.savefig('/home/anamariani/Documents/Resultados/figuras_teste/wavelets/wavelet_tiw_alt_lat{}_lon{}_plot_onlycbar_tiws.png'.format(lat, lons[la]),dpi=300,bbox_inches='tight')

#Mutiplico a variancia para colocar de volta a unidade de Potencia (soma do ipower deve ser igual à variância)
