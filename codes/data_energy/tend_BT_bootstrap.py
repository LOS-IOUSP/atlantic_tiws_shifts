#!/home/ana/anaconda3/bin/python3
# *- coding: utf-8 -*-
"""
Created on Tue Nov 14 14:09 2023

@author: anamariani - adaptado de @polito

Programa que calcula a tendencia da integral da potencia do
espectro do nível do mar onde se encontram as ondas de 
20 a 50 dias para cada ponto de grade da minha área de estudo
"""


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

def plot_map():
    #Plot map with maximum correlation values and maximum lags
    print('Plotando figura')
    rc('font', family={'family': 'serif', 'serif': ['Times']})
    rc('text', usetex=True)
    rc('font', size=12)  # controls default text size
    #plt.ion()
    lon2, lat2 = np.meshgrid(lons,lats)

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
    

    ax.set_extent([lons.min(), lons.max(), -8, 8])
    g = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, 
        color='gray', alpha=0.5, linestyle='--')
    g.top_labels= False
    g.right_labels = False

    #colorbar
    CB = plt.colorbar(hcv, ax=ax, fraction =0.006, aspect = 45, label=r'Trend (Slope) ['+r'$\mu W m^{-3}$/yr'+']',pad=0.03, format='%.1e', ticks=np.linspace(mini, maxi, 5)) #,extend='both')
    CB.set_ticks(np.linspace(mini,maxi,7))

    plt.subplots_adjust(left=0.05, right=0.85, bottom=0.05, top=0.95, wspace=0.05)


    return fig, ax

import numpy as np
import xarray as xr
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.dates as mdates
#from matplotlib.dates import YearLocator, DateFormatter, date2num
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean as cmo

print('Reading variables...')
#Loading dataset
f = xr.open_dataset('/data/anamariani/resultados_mestrado/energy_GLORYS/Barotropic_energy.nc')
f = f.sel(longitude=slice(-60.,15.))
lons = f.longitude.values
lats = f.latitude.values
depth = f.depth.values
time = f.time.values
faket = np.arange(1,len(time)+1,1)

for i_d, d in enumerate(depth):
    print('DEPTH: {:.0f} - {}/{}'.format(d,i_d+1,len(depth)))
    BT = f.BT_mean.sel(depth=d, method='nearest')*1000000 #microWm⁻³
    nt = len(time)

    #onde será colocada a tendencia
    tend_xy = np.ones((len(lats),len(lons)))*-999
    mask_t = np.zeros((len(lats),len(lons))) #Mascara da significancia --> se for 0: não significativo / se for 1: significativo
    mean_xy = np.ones((len(lats),len(lons)))*-999

    for i,lat in enumerate(lats):
        for j,lon in enumerate(lons):
            z = BT.sel(latitude=lat,longitude=lon).values
            if np.any(np.isnan(z)) == False:
                #Calculando tendência pelo ajuste linear
                coef = np.polyfit(faket, z, 1)
                zfit = coef[0]*faket + coef[1]
                slope = coef[0]
                intercept = coef[1]

                #print(intercept)
                tend_xy[i,j] = slope
                mean_xy[i,j] = intercept

                #Calculando significancia pelo método bootstrap
                up_CL, low_CL = bootstrap(z,faket)

                if slope > up_CL or slope < low_CL:
                    mask_t[i,j] = 1.

        print('Latitude {:.3f} ({}/{}) ok!'.format(lat, i, len(lats)))

    print('Fim do calculo das tendências')

    #mascarando dados invalidos
    tend_xy = np.ma.masked_where(tend_xy == -999, tend_xy)
    tend_xy = np.ma.masked_invalid(tend_xy)
    tend_xy.fill_value = np.nan

    print('Salvando <3')
    np.savez('/data/anamariani/resultados_mestrado/energy_GLORYS/mapa_tendencia_BT_D{:.0f}.npz'.format(d), lon=lon, lat=lat, tend_xy= tend_xy, mask_t=mask_t,  
        mean_xy=mean_xy)

    fig, ax = plot_map()

    fig.savefig('/home/anamariani/Documents/Resultados/figuras_teste_GLORYS/mapa_tendencia_BT_D{:.0f}.png'.format(d), bbox_inches='tight', dpi=300)

    print('Salvo!')
