#!/home/ana/anaconda3/bin/python3
# *- coding: utf-8 -*-
"""
Created on Tue Nov 14 14:09 2023

@author: anamariani - adaptado de @polito

Programa que calcula a tendencia da integral da potencia do
espectro do nível do mar onde se encontram as ondas de 
20 a 50 dias para cada ponto de grade da minha área de estudo
"""

def get_aux():
    # red auxiliary data (lat,lon,time) from one of the netcdf files
    fn = '/data/anamariani/Data/altimetria/sealevel_glo_phy_climate_l4_my_008_057/alt_l4_c3s_climate_1993.nc'
    fn2 = '/data/anamariani/Data_pre_proc/altimetria/sealevel_glo_phy_climate_l4_my_008_057_por_lat/alt_l4_c3s_climate_lat4.125_1993_2022.nc'

    f = xr.open_dataset(fn)
    #f = f.sel(latitude=slice(0.,7.5))
    lat = f.latitude.values
    lon = f.longitude.values


    f = xr.open_dataset(fn2)
    #f = f.sel(time=slice('01-01-1993', '31-12-2021'))#'31-12-2021'))
    time = f.time.values
    faket = np.arange(0,len(time),1)
 
    return time, faket, lat, lon


def bootstrap(peaks,time):
    #numero de pontos que eu tenho
    n = len(peaks)
    index = np.arange(0,n,1)

    #numero de perturbações aleatórias que eu quero fazer
    n_perm = 1000

    #matrizes para alocar as permutações
    rand_perm = np.ones((n_perm,n))*np.nan
    r_trend = np.ones(n_perm)*np.nan

    #indice para garantir que não tenham permutações repetidas
    nn=0

    #Loop para calcular a tendência de diferentes permutações
    #dos pontos (fazer diferentes amostragens, sem repetir ponto
    #e sem repetir permutações)
    while nn < n_perm:
        #fazendo a permutação dos pontos
        perm = np.random.permutation(index)
        y = peaks[perm]

        # Verificar se a permutação já foi gerada
        if np.any(np.all(perm == rand_perm, axis=1)) == False:
            rand_perm[nn,:] = perm
            p = np.polyfit(time, y, 1)
            r_trend[nn]= p[0]
            nn += 1

    #Nível de confiânça de 95%
    tm = np.nanmean(r_trend)
    ts = np.nanstd(r_trend)
    up_CL = tm + 2*ts
    low_CL = tm - 2*ts

    return up_CL, low_CL

def pwr_cwt(znd,dt,s0,dj,J):
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

    np.int = np.int_ #desatualizado no pacote
    wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(znd, dt, dj, s0, J, mother)

    periods = 1 / freqs
    #rectify the power spectrum according 
    #to the suggestions proposed by Liu et al. (2007)
    power = (np.abs(wave)) ** 2
    power /= scales[:, None]
    power /= power.sum()

    cc, pp = np.meshgrid(coi, periods)
    mpower = np.ma.masked_where(pp > cc, power)

    ipower = np.sum(mpower.data, axis=0)
    #ipower = ipower/ipower.max()

    return ipower

def plot_map():
    #Plot map with maximum correlation values and maximum lags
    print('Plotando figura')
    rc('font', family={'family': 'serif', 'serif': ['Times']})
    rc('text', usetex=True)
    rc('font', size=12)  # controls default text size
    plt.ion()
    lon2, lat2 = np.meshgrid(lon,lat)

    #Propriedades para plot em cartopy
    pro = ccrs.PlateCarree(lon.mean())
    geo = ccrs.PlateCarree() 
    land_resolution = '10m'
    land_poly = cfeature.NaturalEarthFeature('physical', 'land', 
        land_resolution, edgecolor='k', facecolor=cfeature.COLORS['land'])

    #Vamos plotar a primeira figura
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(8, 3), subplot_kw={'projection': pro})
    mycmap = plt.cm.nipy_spectral #estabelecendo cores do grafico
    m = tend_xy.mean()
    s = tend_xy.std()
    mini = m - 3.5*s
    maxi = m + 2*s
    #mini = 0.001855
    #maxi = 0.00367
    print('Figura:')
    print(m)
    print(s)
    print(tend_xy.max())
    print('mini = {}'.format(mini))
    print('maxi = {}'.format(maxi))

    plt.cla() #limpar

    plt.sca(ax)

    ax.coastlines(resolution=land_resolution, color='black', linewidth=0.5)
    ax.add_feature(land_poly)

    hcv = ax.contourf(lon2, lat2, tend_xy, levels = np.linspace(mini,maxi,30), cmap = mycmap, transform=geo, extend='both') #levels = np.linspace(mini,maxi,30)
    c1 = ax.contour(lon2, lat2, tend_xy, levels = np.linspace(mini,maxi,30), colors='k', alpha=0.2, linewidths=0.7, transform=geo) #linewidths= 1.2
    ax.pcolor(lon2, lat2, np.ma.masked_where(mask_t>0.5,mask_t), hatch='x',alpha=0, shading='nearest', transform=geo)
    #label = plt.clabel(c1, fontsize=7, inline = 3, fmt = '%i mes', inline_spacing = 2, rightside_up = True)
    

    ax.set_extent([lon.min(), lon.max(), -8, 8])
    g = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, 
        color='gray', alpha=0.5, linestyle='--')
    g.top_labels= False
    g.right_labels = False

    #colorbar
    CB = plt.colorbar(hcv, ax=ax, fraction =0.006, aspect = 45, label=r'Trend (Slope) [W/yr]',pad=0.03, format='%.1e', ticks=np.linspace(mini, maxi, 5)) #,extend='both')
    CB.set_ticks(np.linspace(mini,maxi,7))

    plt.subplots_adjust(left=0.05, right=0.85, bottom=0.05, top=0.95, wspace=0.05)


    return fig, ax

import glob
import numpy as np
import xarray as xr
import datetime as dt
from scipy.signal import detrend
import pycwt as wavelet
from scipy.signal import find_peaks
from numpy.linalg import inv
from scipy.signal import correlate, blackman, convolve
from scipy.signal import oaconvolve
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.dates as mdates
#from matplotlib.dates import YearLocator, DateFormatter, date2num
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean as cmo
from statistics import median


print('Reading variables...')
g = glob.glob('/data/anamariani/resultados_mestrado/dados_por_lat_filtrados/semi-auto_new/*.nc')
g.sort()
print(g)

time, faket, lat, lon = get_aux()
time2 = [dt.date(1993, 1, 1) + dt.timedelta(days=int(tt)) for tt in faket]
jtime = [dt.date.toordinal(time2[j]) for j in range(len(time))]
nt = len(jtime)

fill = np.ones((len(time), len(lat), len(lon)))*np.nan
p_fill = np.ones((len(time), len(lat), len(lon)))*np.nan

#onde será colocada a tendencia
tend_xy = np.ones((len(lat),len(lon)))*-999
mask_t = np.zeros((len(lat),len(lon))) #Mascara da significancia --> se for 0: não significativo / se for 1: significativo
mean_xy = np.ones((len(lat),len(lon)))*-999

#componente de 20 a 55 dias
x = np.arange(20,55)
c30 = np.array(['z_R_{}'.format(i) for i in x])

#----------------- wavelet params -------------------------------
# temporal resolution in days
dt = 1.0
s0 = 5.71 #11.048738 #3 Starting scale, in this case ~10 days, fine
# tuned to produce one harmonic near 365.2425 days
dj = 1.0 / 8  # sub-octaves per octave # 1.0/8
J = 6.0 / dj  # 6 powers of two with dj sub-octaves #6.0/dj

for ig, gg in enumerate(g):
    f = xr.open_dataset(gg)
    #f = f.sel(time=slice(jtime[0],jtime[-1]))
    ny = np.where(lat == f.lat.values) [0][0]
    glons = f.lon.values

    keys = np.array(list(f.keys()))
    comp = np.intersect1d(keys, c30)

    if len(comp) != 0:
        for il, glon in enumerate(glons):
            nx = np.where(lon == glon)[0][0]
            z = f[comp[0]].sel(lon=glon).values

            #Subtituindo os valores Nan pontuais na série pela média da série
            if len(z[np.isnan(z)]) > 0:
                print('TEM {:d} NANs em: {:.2f} (lat) {:.2f} (lon)'.format(len(z[np.isnan(z)]),f.lat.values,glon))
                m = np.ma.masked_invalid(z)
                z[m.mask] = np.nanmean(z)

            # normalize
            z_std = z.std()
            z_mean = z.mean()
            z_var = z_std**2
            zn = (z-z_mean)/z_std
            znd = detrend(zn)

            #Calculando a integral da potencia do espectro
            ipower = pwr_cwt(znd,dt,s0,dj,J)
            #multiplicando pela variância de volta
            ipower = ipower*z_var
            fill[:,ny,nx] = ipower
            #Selecionando os picos dessa integral, maiores que a mediana dos dados
            p,_ = find_peaks(ipower)
            mediana = median(ipower)
            p5 = [ip for ip in p if ipower[ip] > mediana]
            peaks = ipower[p5]
            tpeaks = faket[p5]
            p_fill[p5,ny,nx] = peaks

            #Calculando tendência pelo ajuste linear
            coef = np.polyfit(tpeaks, peaks, 1)
            zfit = coef[0]*faket + coef[1]
            slope = coef[0]
            intercept = coef[1]

            #print(intercept)
            tend_xy[ny,nx] = slope*365
            mean_xy[ny,nx] = intercept

            #Calculando significancia pelo método bootstrap
            up_CL, low_CL = bootstrap(peaks,tpeaks)

            if slope > up_CL or slope < low_CL:
                mask_t[ny,nx] = 1.

    print('Latitude {:.3f} ({}/{}) ok!'.format(f.lat.values, ig, len(g)))

print('Fim do calculo das tendências')

#mascarando dados invalidos
tend_xy = np.ma.masked_where(tend_xy == -999, tend_xy)
tend_xy = np.ma.masked_invalid(tend_xy)
tend_xy.fill_value = np.nan

print('Salvando <3')
np.savez('/data/anamariani/resultados_mestrado/wavelets/mapa_tendencia_Pot_tiw_com_var_bootstrap_method_new.npz', lon=lon, lat=lat, time=faket, tend_xy= tend_xy, mask_t=mask_t,  
    mean_xy=mean_xy, peaks=p_fill, ipower=fill)

fig, ax = plot_map()

fig.savefig('/home/anamariani/Documents/Resultados/figuras_teste/wavelets/mapa_tendencia_Pot_tiw_com_var_bootstrap_method_new.png', bbox_inches='tight', dpi=300)

print('Salvo!')
