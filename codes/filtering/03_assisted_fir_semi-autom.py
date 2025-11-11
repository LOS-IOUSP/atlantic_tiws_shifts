#!/home/ana/anaconda3/bin/python3
# *- coding: utf-8 -*-
"""
Created on Wed May 27 09:00:04 2020

@author: paulo

Modified version by anamariani for the application of the filter in the altimeter
data in the Tropical Atlantic region, between 10°N e 10°S
"""


import numpy as np
import pandas as pd
from fir import maxvar, varexp, f_longterm, f_annual, f_semiannual,\
    get_zg, f_wave, get_cLTA, get_aux, get_basin,\
    get_gparms, f_eddies, d2r, acorr
from class_fft import FFT
from class_filter import filter_wave, manual_filter
from collections import OrderedDict as od
from xarray import Dataset
from scipy.interpolate import interp1d, griddata
from scipy.fft import fft, fftfreq, fftshift
from scipy.signal import detrend, find_peaks
from scipy.stats import mode
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.dates import YearLocator, DateFormatter, date2num
from matplotlib.ticker import (StrMethodFormatter, MultipleLocator)
import cmocean.cm as cmo
import glob


def save_nc():
    '''
    Saves the output file, variables are specific to this program
    '''
    outfile = '/home/anamariani/Documents/Resultados/resultados_mestrado/dados_por_lat_filtrados/semi-auto/hovs_filt_{:}_{:05.3f}_filtro_semi-auto_1.nc'.format(bas, lat)
    print('[save_nc] Saving {:}'.format(outfile))
    ddout = {}  # clean ddout just in case
    ddout = {'time': {'dims': 'time', 'data': jtime, 'attrs': {'units': 'd'}},
             'lat': {'dims': [], 'data': lat, 'attrs': {'units': 'deg N/S'}},
             'lon': {'dims': 'lon', 'data': xio, 'attrs': {'units': 'deg E/W'}},
             'bas': {'dims': [], 'data': bas, 'attrs': {'descr': 'basin name'}}}
    encoding = {}

    for key in z_c.keys():
        ddout.update({'z_'+key: {'dims': ('time', 'lon'), 'data': z_c[key],
                                 'attrs': {'units': 'mm', 'descr': 'sea level anomaly filtered'}}})
        encoding.update({'z_'+key: {'dtype': 'int16', 'scale_factor': 0.01, 'zlib': True,
                                    '_FillValue': -9999}})
        ddout.update({'A_'+key: {'dims': [], 'data': A[key],
                                 'attrs': {'units': 'mm', 'descr': 'amplitude'}}})
        encoding.update({'A_'+key: {'dtype': 'int16', 'scale_factor': 0.01,
                                    'zlib': True, '_FillValue': -9999}})
        ddout.update({'vx_'+key: {'dims': [], 'data': vx[key],
                                  'attrs': {'units': 'unitless', 'descr': 'variância explicada em relação ao sinal original'}}})
        encoding.update({'vx_'+key: {'dtype': 'int16', 'scale_factor': 0.01,
                                     'zlib': True, '_FillValue': -9999}})
    for key in L.keys():
        ddout.update({'L_'+key: {'dims': [], 'data': L[key], 'attrs': {'units': 'km',
            'descr': 'comprimento de onda'}}})
        encoding.update({'L_'+key: {'dtype': 'int16', 'scale_factor': 1.0,
                                    'zlib': True, '_FillValue': -9999}})
        ddout.update({'Le_'+key: {'dims': [], 'data': Le[key], 'attrs': {'units': 'km',
            'descr': 'erro associado à estimativa do comprimento de onda'}}})
        encoding.update({'Le_'+key: {'dtype': 'int16', 'scale_factor': 1.0,
                                     'zlib': True, '_FillValue': -9999}})
        ddout.update({'T_'+key: {'dims': [], 'data': T[key], 'attrs': {'units': 'km', 
            'descr': 'período da onda'}}})
        encoding.update({'T_'+key: {'dtype': 'int16', 'scale_factor': 0.01,
                                    'zlib': True, '_FillValue': -9999}})
        ddout.update({'Te_'+key: {'dims': [], 'data': Te[key], 'attrs': {'units': 'd', 
            'descr': 'erro associado ao período da onda'}}})
        encoding.update({'Te_'+key: {'dtype': 'int16', 'scale_factor': 0.01,
                                     'zlib': True, '_FillValue': -9999}})
        ddout.update({'cp_'+key: {'dims': [], 'data': cp[key], 'attrs': {'units': 'km', 
            'descr': 'velocidade de fase'}}})
        encoding.update({'cp_'+key: {'dtype': 'int16', 'scale_factor': 0.01,
                                     'zlib': True, '_FillValue': -9999}})
        ddout.update({'cpe_'+key: {'dims': [], 'data': cpe[key], 'attrs': {'units': 'km', 
            'descr': 'erro associado à velocidade de fase'}}})
        encoding.update({'cpe_'+key: {'dtype': 'int16', 'scale_factor': 0.01,
                                      'zlib': True, '_FillValue': -9999}})
    ds = Dataset.from_dict(ddout)
    ds.to_netcdf(outfile, format='NETCDF4', encoding=encoding)
    return ds
# -----------------------------------------------------------------------------



def write_report():
    '''
    Write a little report, variables are specific to this program
    '''
    for key in z_c.keys():
        print('------------------------ write_report -----------------------------')
        if key[0:2] == 'R_':
            print('{:} -> cp=({:5.1f} +- {:5.1f})km/d,'.format(key, cp[key], cpe[key]))
            print('L=({:.0f} +- {:.0f})km,'.format(L[key], Le[key]))
            print('T=({:.0f} +- {:.0f})d,'.format(T[key], Te[key]))
            print('A={:.0f}mm'.format(A[key]))
        else:
            print('{:} -> A={:5.1f}mm'.format(key, A[key]))
        print('vx={:.0f}%'.format(vx[key]))
        print('--------------------------------------------------------------------')
# -----------------------------------------------------------------------------


def plot_hovs():
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

def select_basin(bas, y, lat, beta):
    # mask number of the basin (0 -> continent)
    io = 1 + np.where(np.array(basin) == bas)[0][0]
    nf = np.where(y == lat)[0][0]
    # slice according to latitude index
    xb = mask[nf, :] #longitude da mascara na latitude selecionada
    # find indices of xb that correspond to this basin (io)
    ib = np.where(xb == io)[0]
    # crop longitudes, Rossby radii and gravity wave phase speeds
    # to current basin
    xio = x[ib] #longitudes da area de estudo que estão dentro da bacia e latitude selecionada (sem continentes)
    #nl = np.where(xio >= lon)[0][0]
    #xio = xio[nl:]
    #print('[select_basin] xio = {}, nl={}'.format(xio, nl))

    # check if there is enough data to be processed
    nxio = len(xio)
    if nxio <= minxlen:
        print('[select_basin] 1 - Narrow basin ({:} on the {:}), nothing to do.'.format(lat,
              bas.capitalize()))
        return ([-9999], 0)
    # crop SSHA to the current basin
    # and set masked z values to z.fill_value
    z = zg[:, ib].filled()
    #z = z[:,nl:]
    # fix mismatch between continental mask and cmems data
    zm0 = z.mean(axis=0)
    mm0 = np.where(zm0 != zg.fill_value)[0] #pontos em que tem dado (não é continente)
    # check if enough data remains to be processed
    if len(mm0) <= minxlen:
        print('[select_basin] 2 - Narrow basin ({:} on the {:}), nothing to do.'.format(lat,
              bas.capitalize()))
        return ([-99999], 0)
    # crop the empty edges
    i0 = np.min(mm0)
    i1 = 1 + np.max(np.where(zm0 != zg.fill_value))
    if (i0 != 0) or (i1 != nxio): #conferindo se os indices dos continentes na mascara corresponde aos do dado real e corrigindo
        z = z[:, i0:i1]
        xio = xio[i0:i1]
        nxio = len(xio)
    # check if enough data remains to be processed
    if nxio <= minxlen:
        print('[select_basin] 3 - Narrow basin ({:} on the {:}), nothing to do.'.format(
              lat, bas.capitalize()))
        return ([-99999], 0)
    return (z, xio)

# -----------------------------------------------------------------------------


def filter_z(z):
    tn = np.arange(len(time))*delt
    xn = np.arange(len(xio))*delx
    # Initialize all output as ordered dictionaries - od: dicionario que lembra da ordem de incerção dos elemntos
    z_c, L, Le, T, Te, cp, cpe, A, vx = od(), od(), od(), od(), od(), od(), od(), od(), od()
    # we will operate on z, save the originalz
    # test data: substitute real data by sinusoidal wave or a random data
    just_for_test = False
    if just_for_test is True:
        xx, tt = np.meshgrid(111*xio-xio[0], np.arange(len(time)))
        #AA = 100
        #z = AA*np.sin((2*np.pi/2000)*xx + (2*np.pi/66)*tt)
        a = np.random.randn(z.shape[0], z.shape[1])
        m = z.mean()
        s = z.std()
        print('Z: Media - {:.2f}, Std - {:.2f}'.format(m,s))
        z = a*s + m
        print('Z: Media - {:.2f}, Std - {:.2f}'.format(z.mean(),z.std()))

    z_c['ori'] = z.copy()
    # assume sinusouidal signal, amplitude is sqrt(2)*std()
    A['ori'] = np.sqrt(2)*np.std(z_c['ori'])
    # get trend and very low frequency non-propagating signal
    z_c['longterm'] = f_longterm(z, ny=7, delt=delt) #7 anos - máxima variabilidade do El Nino (Variabilidade parecida com Atlantic Nino)
    z_c['longterm'] = maxvar(z_c['longterm'], z)*z_c['longterm'] #"compensar algum eventual excesso de suavização imposto pelo filtro"
    A['longterm'] = np.sqrt(2)*np.std(z_c['longterm'])
    z = z_c['ori'] - z_c['longterm']
    print('[filter_z] Long term signal extracted.')
    # get annual (=seasonal) signal as a basin-wide bump with annual frequency
    z_c['annual'] = f_annual(z, delt)
    z_c['annual'] = maxvar(z_c['annual'], z)*z_c['annual']
    A['annual'] = np.sqrt(2)*np.std(z_c['annual'])
    z = z - z_c['annual']
    print('[filter_z] Annual signal extracted.')

    z_exp = z.copy()

    keep_filtering = True
    while keep_filtering:
        fft = FFT(z, delt, delx, lat)
        n = int(input('Quantos picos quer identificar na Transformada de Fourrier?'))
        fig, ax, mft, mfx = fft.show_FFT_mean(n)
        print('Você quer selcionar alguma componente com esses períodos [p]? Quer filtrar manualmente[m]? Quer parar [s]? Ou ja tem uma componente feita [c]? \n')
        for ii in range(n):
            print('{} - {:.0f} dias \n'.format(ii, 1/mft[ii]))
        s = str(input('[p/m/s/c]'))
        if s == 'p':
            T1 = float(input('Qual o periodo em dias? '))
            #T1 = 1/mft[com]
            keep_tryng = True
            while keep_tryng:
                #ss = str(input('Quer tentar o L médio: {:.0f}km? [s/n] '.format(1/mfx[com])))
                #if ss == 's':
                #    L1 = -1/mfx[com]
                #else:
                    #fig2, ax2 = fft.show_hist(4, com)
                Test = str(input('Quer testar diferentes L? [s/n]'))
                if Test == 's':
                    L1 = float(input('Sugira o primeiro L que deseja testar em km: \n'))*(-1)
                    L2 = float(input('Sugira o ultimo L que deseja testar em km: \n'))*(-1)
                    I = int(input('Sugira o intervalo que deseja testar em km: \n'))*(-1)
                    L_t = np.arange(L1,L2,I)
                    var_teste = np.zeros(len(L_t))
                    for u, ll in enumerate(L_t):
                        fwave = filter_wave(delx, delt, lat, z, T1, ll, z_c['ori'])
                        var_teste[u] = fwave.vari_expl()
                    print('Variancia Explicada')
                    print(var_teste)
                    print('Comprimentos de Onda')
                    print(L_t)
                    plt.figure()
                    plt.plot(L_t, var_teste)
                    plt.pause(10)
                    print('O maximo L foi de: {}'.format(L_t[var_teste == var_teste.max()]))
                L1 = float(input('Digite o L escolhido em km: \n'))*(-1)
                fwave = filter_wave(delx, delt, lat, z, T1, L1, z_c['ori'])
                fig3, ax3 = fwave.show_hovs(tn, xn)
                res = z - fwave.matriz_filtro()
                fft_r = FFT(res, delt, delx, lat)
                fig4, ax4, mft2, mfx2 = fft_r.show_FFT_mean(n)
                print('Periodos')
                print(1/mft2)
                print('Comprimentos de Onda')
                print(1/mfx2)
                bom = input('Ficou bom? [s/n]')
                if bom == 's':
                    if L1 < 0:
                        Rc = 'R_' + str(int(T1))
                    else:
                        Rc = 'K_' + str(int(T1))
                    z_c[Rc] = fwave.matriz_filtro()
                    cp[Rc], cpe[Rc], L[Rc], Le[Rc], T[Rc], Te[Rc], A[Rc] = fwave.get_param()
                    z = z - z_c[Rc]
                    keep_tryng = False
        elif s == 'm':
            fwave_m = manual_filter(delx, delt, lat, z, z_c['ori'], tn, xn)
            fig, ax = fwave_m.show_hovs(tn, xn)
            bom = input('Ficou bom? [s/n]')
            if bom == 's':
                cp1, T1 = fwave_m.retn_cp()
                if cp1 < 0:
                    Rc = 'R_' + str(int(T1)) #Se a velocidade de fase é negativa, é uma onda de Rossby
                else:
                    Rc = 'K_' + str(int(T1))
                z_c[Rc] = fwave_m.matriz_filtro()
                z = z - z_c[Rc]
                cp[Rc], cpe[Rc], L[Rc], Le[Rc], T[Rc], Te[Rc], A[Rc] = fwave_m.get_param()
                df = fwave_m.df_selected()
                df.to_csv('/data/anamariani/resultados_mestrado/dados_por_lat_filtrados/semi-auto/param/periods_lat{}_componet_{}.csv'.format(lat,Rc))
        elif s == 'c':
            arq = input('Digite o nome do arquivo:\n')
            par = pd.read_csv('/data/anamariani/resultados_mestrado/dados_por_lat_filtrados/semi-auto/param/'+arq)
            T1 = par['T filter'][0]
            L1 = par['L filter'][0]
            print('[filter_z] --- supervised parameters ---')
            print('[filter_z] T1=({:.1f}) days'.format(T1))
            print('[filter_z] L1=({:.0f}) km\n'.format(L1))
            if par['Cp selected'].mean() < 0:
                Rc = 'R_' + str(int(T1))
            else:
                Rc = 'K_' + str(int(T1))

            fwave = filter_wave(delx, delt, lat, z, T1, L1, z_exp)
            z_c[Rc] = fwave.matriz_filtro()

            cp[Rc], cpe[Rc], L[Rc], Le[Rc], T[Rc], Te[Rc], A[Rc] = fwave.get_param()
            print('[filter_z] --- unsupervised parameters ---\n',
            '[filter_z] Equatorial wave component ',
            '[{:}] found, T0 = {:.0f} days,\n'.format(Rc, T[Rc]),
            '[filter_z] L0={:.0f} km, and cp={:.1f} km/d.'.format(L[Rc], cp[Rc]))
            fvar = 100*varexp(z_exp, z_c[Rc])
            print('[filter_z] z[{:}] explains {:.0f} % of the input variance.'.format(Rc, fvar))
            z = z - z_c[Rc]
            sn_c = input('Você tem outra componente ja feita? [s/n]\n')
            print(z_c)
        elif s == 's':
            keep_filtering = False

        else:
            s = str(input('Opção inválida, digite novamente [p/m/s/c]: '))

    z_c['residual'] = z.copy()
    A['residual'] = np.sqrt(2)*np.std(z_c['residual'])

    for key in z_c.keys():
    	vx[key] = 100*varexp(z_c['ori'], z_c[key])

    return z_c, L, Le, T, Te, cp, cpe, A, vx
            
    # ==================================================================================

# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------




#Path of the altimeter data files divided by latitude
path = '/data/anamariani/Data_pre_proc/altimetria/sealevel_glo_phy_climate_l4_my_008_057_por_lat/'
files = glob.glob(path + '*.nc')

#List with all the latitudes to be processed
#lats = [f[-19:-13] for f in files]
#lats = sorted(list(map(organize_lats, lats)))

#For interative input
#llat = float(input('Quais latitudes quer fazer?'))
#lats = [llat,]

lats = [4.625, ]
#lon = -31.0

# basins to be processed
basins = ['atlantic', ]
# minimum number of points in the longitudinal direction worth processing
minxlen = 25

delt = 1     # temporal grid spacing in days
delx = 0.25  # longitudinal grid spacng in degrees

# -----------------------------------------------------------------------------
# read coordinates and shapes
nx, ny, nt, x, y, time, jtime = get_aux(files[0])
# load basin mask
mask, basin = get_basin()

# big loops
for lat in lats:
    # Parameters
    f0, beta, Omega, Re = get_gparms(lat)
    for bas in basins:
        plt.close('all')
        # load global data 2D array to be filtered
        zg = get_zg(lat, path) #pega altura da superfície do dado em .nc e converte para mm
        print('[main] Processing lat={:} of the {:}'.format(lat, bas.capitalize()))
        (z, xio) = select_basin(bas, y, lat, beta)
        if np.isnan(z).any():
            print(np.where(np.isnan(z)==True))
# =============================================================================
        z_c, L, Le, T, Te, cp, cpe, A, vx = filter_z(z)


        # print the parameters
        write_report()
#
        # Saving
        #ds = save_nc()
#
        # Plotting
        #fig, ax0 = plot_hovs()
        #print('Salvando a figura...')
        #fig.savefig('/home/anamariani/Documents/Resultados/figuras_teste/filtro/semi-auto/teste_hovs_latitude{}_filtro_semi-auto_1.png'.format(lat), dpi=300)
        #print('Salvo <3')
#
# =============================================================================
