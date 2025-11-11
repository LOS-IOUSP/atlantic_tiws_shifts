#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 15:04:10 2020

@author: paulo
"""
from numpy import mod, linspace, exp, meshgrid, pi, sqrt, sin, cos, tan,\
                  newaxis, round, var, sign, sum, convolve, arctan2, where,\
                  ones, zeros, mean, std, nonzero, concatenate, arange, diff,\
                  asarray, array, load, isnan, nan
import numpy.ma as ma
from numpy.linalg import lstsq
from scipy.signal import oaconvolve, blackman, find_peaks
from scipy.fft import fft, fftfreq, fftshift
from scipy.io import loadmat
from scipy.stats import mode
from skimage.transform import radon, rescale
from netCDF4 import Dataset
import datetime as dt

# -----------------------------------------------------------------------------

#convert radianos to degrees
def r2d(a):
    return 180*a/pi
# -----------------------------------------------------------------------------

#convert degrees to radianos
def d2r(a):
    return pi*a/180
# -----------------------------------------------------------------------------

def organize_lats(a):
    '''
    Organize a list os lats based on the latitude
    in the title of the file
    a: string
    return a float
    '''
    if 't' in a:
        return float(a[1:])
    else:
        return float(a)
#--------------------------------------------------------------------------------

def gauss(nx, nt, s):
    ''' g, x0, y0 = gauss(nx, nt, s)
        Returns g, a 2D Gaussian kernel with nx columns and nt rows; s is
        the standard deviation and thus controls the shape of the curve.
        nx and nt should be odd integers. Returns also the x0, y0
        pixel grid space'''
    nx = int(nx) #lines
    #garantir que nx e ny sejam ímpar, uma vez que o tamanho do kernel gaussiano deve ser ímpar para ter um centro.
    if mod(nx, 2) == 0: 
        nx += 1
    nt = int(nt) #columns 
    if mod(nt, 2) == 0:
        nt += 1
    # (?) - número de linhas e colunas deve ser ímpar?
    # create a mesh of shape (v,h) - grid 2D de coordenadas x0 e y0
    x0, y0 = meshgrid(linspace(0.5, nx-0.5, num=nx),
                      linspace(0.5, nt-0.5, num=nt))
    g = exp(-0.5*((pi*(x0-nx/2)/(nx*s))**2 + (pi*(y0-nt/2)/(nt*s))**2))\
        / (sqrt(2*pi) * s)
    # sum should be one, there are no negative values
    g = g-g.min() #garante que não tem numeros negativos
    g = g/g.sum() #garante que soma dos valores deve ser 1
    return g, x0, y0
# -----------------------------------------------------------------------------


def sombrero(nx, nt, L, T):
    ''' sf = sombrero(nx, nt, L, T)
        Returns sf, a 2D tapered cosine kernel with nx columns and nt rows.
        If rows and columns represent space and time, then L and T are the
        wavelength and period that are used in the cosine function. Tapering
        at the borders is achieved by scalar multiplication with the Gaussian
        gauss(nx, nt). Note that nx, nt, T, L are all in pixel units'''
    s = 1
    g, x0, y0 = gauss(nx, nt, s)
    x1 = x0-x0.mean()
    y1 = y0-y0.mean()
    # empirically compensate the fact that the multiplication by the
    # gaussian in practice shrinks T and L.
    c = 0.85
    sf = g*cos((c*2*pi/L)*x1 - (c*2*pi/T)*y1)
    # sum of abs should be one, integral over x,t should be zero
    sf = sf-sf.mean()
    sf = sf/sum(abs(sf))
    return sf,x0,y0
# -----------------------------------------------------------------------------


def obconvolve(z, f):
    ''' obconvolve(z,f) one-liner based on scipy.signal.oaconvolve
    Returns z1 which is the  result of the convolution of z and f, with borders
    rectified by dividing z by the convolution of a unitary matrix shaped as z
    with the kernel f. Mode='same' is hardwired.
    '''
    z0 = oaconvolve(z, f, mode='same') / oaconvolve(ones(z.shape), abs(f),
                                                    mode='same')
    return z0
# -----------------------------------------------------------------------------


def acorr(z):
    ''' acorr(z) one-liner based on scipy.signal.oaconvolve
    Returns z1 which is the autocorrelation of z, with borders
    rectified by dividing z by the convolution of a unitary matrix shaped as z
    with itself. Mode='same' is hardwired.
    '''
    z0 = oaconvolve(z, z, mode='same') / oaconvolve(ones(z.shape),
                                                    ones(z.shape), mode='same')
    z0 = -z0[::-1, ::-1]
    return z0

# -----------------------------------------------------------------------------


def maxvar(a, b):
    ''' m = maxvar(a,b) returns the scalar m that minimizes the variance of
    (a - m*b)
    '''
    a = a - a.mean()
    b = b - b.mean()
    a1 = a.flatten()[newaxis]
    b1 = b.flatten()[newaxis]
    m, _, _, _ = lstsq(a1.T, b1.T, rcond=None)
    return m
# -----------------------------------------------------------------------------


def varexp(a, b):
    ''' v = varexp(a,b) returns the fraction of variance of a explained by b
    '''
    v = 1 - var(a-b)/var(a)
    return v
# -----------------------------------------------------------------------------


def f_longterm(z, ny, delt):
    ''' f_longterm(z) returns the low-pass filtered z that keeps the multi-year
    half-basin-wide trend. ny is the number of years and delt is the temporal
    resolution of z.
    '''
    nx = z.shape[1]
    nt = int(round(ny*365.25/delt))
    if mod(nt, 2) == 0:
        nt += 1
    if mod(nx//2, 2) == 0:
        nx += 2
    g, _, _ = gauss(nx//2, nt, 2) #curva gaussiana com tamanho de metade da bacia (colunas) e 5 anos (linhas) e sigma=2 (esbeltez da gaussiana)
    zf = obconvolve(z, g)
    return zf
# -----------------------------------------------------------------------------


def f_annual(z, delt):
    ''' f_annual(z,delt) returns the band-pass filtered z that keeps the annual
    signal. Uses a semiannual gaussian kernel, with half-basin length.
    '''
    nt = int(round(0.5*365.25/delt))
    if mod(nt, 2) == 0:
        nt += 1
    nx = z.shape[1]
    if mod(nx, 2) == 0:
        nx += 2
    g, _, _ = gauss(nx, nt, 2)
    zf = obconvolve(z, g) #convolução - filtra o valor da função nos pontos em que você tem o impulso
                          #           - valor da função aplicado a um determinado ponto 
                          #           --> somatória de todos os pontos de uma função multiplicada pela resposta a um 
                          #                 determinado impulso - ver onde a função ocorre ou não(?)
                          #           - multiplica o sinal de uma função em um ponto ao sinal da outra função 
                          #                    no mesmo ponto --> soma todos os pontos
                          #             - Média ponderada da imagem ou sinal, utilizando como peso a função kenel, 
                          #              que percorre a função original
                          #           - Objetivo: Destacar ou enfatizar certos padrões ou características 
                          #           do sinal original ou imagem 

    return zf
# -----------------------------------------------------------------------------

def f_semiannual(z, delt):
    ''' f_semiannual(z,delt) returns the band-pass filtered z that keeps the semiannual
    signal. Uses a 3 months gaussian kernel, with half-basin length.
    '''
    nt = int(round(0.25*365.25/delt))
    if mod(nt, 2) == 0:
        nt += 1
    nx = z.shape[1]
    if mod(nx, 2) == 0:
        nx += 2
    g, _, _ = gauss(nx, nt, 2)
    
    zf = obconvolve(z, g)
    return zf
# -----------------------------------------------------------------------------

def f_eddies(z, delt, delx):
    ''' f_eddies(z,delt) returns the band-pass filtered z that keeps the signal from non-propagating mesoscale 
    eddies. Uses a 50 days gaussian kernel, with 5° length.
    '''
    nt = int(round(50/delt))
    if mod(nt, 2) == 0:
        nt += 1
    #nx = z.shape[1]
    #if mod(nx//2, 2) == 0:
    #    nx += 2
    #g, _, _ = gauss(nx//2, nt, 2)
    nx = int(round(5/delx))
    if mod(nx,2) == 0:
        nx += 1
    g, _, _ = gauss(nx, nt, 2)
    #filtro gauss usando comprimento do tamanho da bacia
    zf = obconvolve(z, g)
    return zf
# -----------------------------------------------------------------------------



def f_wave(z, delx, delt, lat, L, T):
    '''
     L in km T in days, L positive is eastward, negative is westward
    '''
    # n sets the size of the filter, empirical tests show that
    # setting n=1.25 is enough to make dyadic filter kernels based on
    # this sombrero function result in orthogonal fields.
    # The larger the n is, the narrower the pass-band will be.
    n = 1.25
    nx = int(round(abs(L)/(delx*111.195*cos(d2r(lat)))))
    nt = int(round(T/delt))
    if mod(nt, 2) == 0:
        nt += 1
    if mod(nx, 2) == 0:
        nx += 1
    print('[f_wave] numero de pontos em x = {}'.format(nx))
    print('[f_wave] numero de pontos em t = {}'.format(nt))
    f = sombrero(n*nx, n*nt, sign(L)*nx, nt)
    zf = obconvolve(z, f)
    return zf
# -----------------------------------------------------------------------------


def tiler(z, L, T, delx, delt, lat):
    '''
    Reassemble z in tiles measuring approximately L by T
     L in km T in days, L positive is eastward, negative is westward.
     Returns tiles, a 4D array measuring [ntilesT,ntilesx,Tinpoints,Linpoints]
    ''' #(?) - não entendi muito bem esse
    nt, nx = z.shape
    #
    L0 = round(abs(L)/(delx*111.195*cos(d2r(lat))))
    ntx = int(round(nx/L0))
    L1 = nx//ntx  # tile width in points
    ex = (nx - L1*ntx)//2  # excess points
    #
    T0 = round(T/delt)
    ntt = int(round(nt/T0))
    T1 = nt//ntt  # tile height in points
    et = (nt - T1*ntt)//2
    #
    tiles = z[et: et + T1*ntt, ex:ex + L1*ntx].reshape((ntt, T1, ntx,
                                                        L1)).swapaxes(1, 2)
    return tiles
# -----------------------------------------------------------------------------

def find_major_peack(st):
    '''
    Encontra o maior pico na série st
    '''
    p,_ = find_peaks(st)
    picos_ordenados = sorted(p, key=lambda x: st[x], reverse=True)
    p1 = picos_ordenados[0]
    return p1

def find_frequency(ff, facf):
    '''
    Encontra o maior pico para cada uma das séries
    na matriz facf e calcula a moda
    '''
    xs = facf.shape[0]
    ys = facf.shape[1]
    f1 = ones(ys)*nan
    facf1 = ones(ys)*nan
    for i in range(ys):
        p1 = find_major_peack(facf[:,i])
        f1[i] = ff[p1]
        facf1[i] = facf[p1,i]
    mf1,_ = mode(f1)
    mfacf1,_ = mode(facf1)
    return mf1[0], mfacf1[0]

def get_cLT2(z, delx, delt, lat):
    #Aplicando a transformada Fourrier na autocorrelação da série 
    #em cada longitude e em cada tempo para encontrar as frequencias e 
    #comprimentos de onda dominantes 
    A = sqrt(2)*z.std()
    (npt, npx) = z.shape
    ac = acorr(z) #autocorrelation of z, with borders rectified by dividing z --> procurar padrões periódicos - ciclos ou oscilações repetitivas nos dados
    # fact = abs(fft(ac, axis=0)).mean(axis=1)
    # facx = abs(fft(ac, axis=1)).mean(axis=0)
    #aplica a transformada de Fourier na autocorrelação para obter as frequências associadas a esses padrões cíclico
    fact = abs(fft(ac, axis=0)) #Para encontrar os períodos predominantes
    facx = abs(fft(ac, axis=1)) #Para encontrar os comprimentos de onda predominantes
    ft = fftfreq(fact.shape[0], d=delt)
    fx = fftfreq(facx.shape[1], d=delx*111.195*cos(d2r(lat)))
    #  use 1 + np* to avoid the DC -  reorganiza as transformadas de Fourier usando a função 'fftshift', movendo a componente DC (frequência zero) para o centro
    fact = fftshift(fact)[1 + npt//2:, :]
    facx = fftshift(facx)[:, 1 + npx//2:]
    ft = fftshift(ft)[1 + npt//2:]
    fx = fftshift(fx)[1+npx//2:]
    #print(imfx.shape)
    #Encontra o maior pico das transformadas de Fourier nas direções do tempo e do espaço
    ft1, fact1 = find_frequency(ft, fact)
    fx1, facx1 = find_frequency(fx, facx.T)

    #Define os períodos e comprimentos de onda
    Td = 1/ft1
    Lkm = 1/fx1

    cp_kmd = Lkm/Td

    return cp_kmd, Lkm, Td, A


def get_cLTA(z, delx, delt, lat):
    '''
    cp_kmd, cp_kmde, L_km, L_kme, T_d, T_de, A = get_cLTA(z, delx, delt, lat)
    Estimates cp (phase speed), L (wave length), and T
    (period) and their respective errors, plus A (amplitude).
    z is the hovmoller-like array, del* are the grid spacings and lat is the
    local latitude. The code is based on the FFT of the autocorrelation.
    '''
    A = sqrt(2)*z.std()
    (npt, npx) = z.shape
    ac = acorr(z) #autocorrelation of z, with borders rectified by dividing z --> procurar padrões periódicos - ciclos ou oscilações repetitivas nos dados
    # fact = abs(fft(ac, axis=0)).mean(axis=1)
    # facx = abs(fft(ac, axis=1)).mean(axis=0)
    #aplica a transformada de Fourier na autocorrelação para obter as frequências associadas a esses padrões cíclico
    fact = abs(fft(ac, axis=0)) #Para encontrar os períodos predominantes
    facx = abs(fft(ac, axis=1)) #Para encontrar os comprimentos de onda predominantes
    ft = fftfreq(fact.shape[0], d=delt)
    fx = fftfreq(facx.shape[1], d=delx*111.195*cos(d2r(lat)))
    #  use 1 + np* to avoid the DC -  reorganiza as transformadas de Fourier usando a função 'fftshift', movendo a componente DC (frequência zero) para o centro
    fact = fftshift(fact)[1 + npt//2:, :]
    facx = fftshift(facx)[:, 1 + npx//2:]
    ft = fftshift(ft)[1 + npt//2:]
    fx = fftshift(fx)[1+npx//2:]
    #Encontra os índices dos máximos das transformadas de Fourier nas direções do tempo e do espaço
    imft = nonzero(fact == fact.max(axis=0))[0]
    imfx = nonzero(facx.T == facx.max(axis=1))[0]


    #Define os períodos e comprimentos de onda
    Tds = 1/ft[imft]
    Lkms = 1/fx[imfx]
    gT = (Tds >= Tds.mean() - Tds.std()) &\
         (Tds <= Tds.mean() + Tds.std())
    #print('[get_cLTA] Períodos:')
    #print(Tds[gT])
    T_d = mean(Tds[gT])
    T_de = std(Tds[gT])
    gL = (Lkms >= Lkms.mean() - Lkms.std()) &\
         (Lkms <= Lkms.mean() + Lkms.std())
    #print('[get_cLTA] Comprimentos de Onda:')
    #print(Lkms[gL])
    L_km = mean(Lkms[gL])
    L_kme = std(Lkms[gL])
    cp_kmd = L_km/T_d
    cp_kmde = cp_kmd*(T_de/T_d + L_kme/L_km)
    # T_d = 1/ft[where(fact == fact.max())][0]
    # L_km = 1/fx[where(facx == facx.max())][0]
    return cp_kmd, cp_kmde, L_km, L_kme, T_d, T_de, A
# -----------------------------------------------------------------------------


def get_cp(z, L, T, delx, delt, lat):
    '''
    Radon transform ---> NOT in use, too slow. See get_cLTA(...)
    '''
    tiles = tiler(z, L, T, delx, delt, lat)
    ntt, ntx, T1, L1 = tiles.shape
    c_pkmd2 = zeros((ntt, ntx))

    for i in range(ntt):
        for j in range(ntx):
            z0 = tiles[i, j, :, :].squeeze()
            # calculate tile autocorrelation, normalize to account for
            # decay at borders
            xc = acorr(z0)
            # increase resolution nr times to avoid quantization nr=3 ->lento
            # nr = 3
            # xc = rescale(xc, nr, anti_aliasing=True)
            # theta are the useful the angles to calculate the Radon transform
            # we don't need to go all the way, from 0 to 180 degrees
            if L > 0:
                theta = linspace(r2d(arctan2(L1, 4*T1)),
                                 r2d(arctan2(L1, 0.25*T1)), 100)
            else:
                theta = linspace(r2d(arctan2(-L1, 0.25*T1)),
                                 r2d(arctan2(-L1, 4*T1)), 100)
            # theta=linspace(-90,90,181)
            # do the transform
            rt = radon(xc, theta=theta, circle=False, preserve_range=True)
            # for rectangular matrices, the transform is angle-dependent,
            # normalize
            rt1 = radon(ones(xc.shape), theta=theta, circle=False,
                        preserve_range=True)
            # mask out the (large number of) zeros
            rt = ma.masked_where(rt == 0, rt)
            # eliminate the corners where we would normalize by a small
            # number of points
            rt1 = ma.masked_where(rt1 < rt1.mean(), rt1)
            # normalize
            rtn = rt/rt1
            # the maximum std is the most reliable angle detector
            rti = rtn.std(axis=0)
            # rescale before smoothing to reduce wiggles at the borders
            rti = (rti-rti.min())/rti.std()
            b = blackman(11)
            b = b/b.sum()
            # smooth it
            rtis = convolve(rti, b, 'same')
            ia = where(rtis == rtis.max())[0][0]
            c_pkmd2[i, j] = tan(d2r(theta[ia]))*delx*111.195*cos(d2r(lat))/delt
    cp_kmd = c_pkmd2.mean()
    cp_kmde = c_pkmd2.std()
    return cp_kmd, cp_kmde


# -----------------------------------------------------------------------------
def get_aux(path):
    # red auxiliary data (lat,lon,time) from one of the netcdf files
    # x from 0.125 to 359.875 y from -89.875 to 89.875
    fn = path #não importa qual seja esse dado (só precisa ser um corte de 
                                                        #eta(x,t) da base de dados de altimetro que está usando) - pois
                                                        #como corta por latitude, as logitudes de todas vão ser a mesma
                                                        #(retangulo)
    ds = Dataset(fn, 'r', decode_times=False)
    x = ds['longitude'][:].data
    nx = x.shape[0]
    y = arange(-66.125, 66.125+0.25, 0.25) #aqui são as latitudes da mascara com as bacias
    ny = y.shape[0]
    # cmems series starts at 1/1/1993
    if diff(ds['time'][:]).max() != 1:
        raise Exception('There should be no gaps in the time series. \
        You probably missed a CMEMS map file.')
    faket = arange(0, len(ds['time'][:].data))
    time = [dt.date(1993, 1, 1) + dt.timedelta(days=int(tt))
            for tt in faket] #time = data na forma datetime
    nt = len(time)
    jtime = [dt.date.toordinal(time[j]) for j in range(nt)] #jtime = data na forma de numeros inteiros
    jtime = asarray(jtime).astype('int32')
    return nx, ny, nt, x, y, time, jtime


# -----------------------------------------------------------------------------
def get_basin():
    # read basin mask for regions deeper than 1000m, small islands removed
    d = loadmat('/home/anamariani/Documents/Resultados/dados_auxiliares/mask025.mat')
    ma = round(d['MM'])
    # add two rows of zeros to make mask the same shape as Rrad
    m0 = zeros((1, ma.shape[1]))
    mask = concatenate((m0, m0, ma, m0, m0), axis=0)
    #como longitude da mask está de 0 a 360 incluindo todo o globo e a longitude dos 
    #meus dados está cortada só para a área do atlantico e no formato de -180 a 180,
    #estou fazendo uma manipulação para cortar e alterar as longitudes da mascara,
    #cortando apenas o pedaço que eu quero.
    #Se for usar dados de todo o globo, comentar essa parte
    ncut = len(d['X'][0,:])//2
    lon_m = concatenate((d['X'][0,ncut:]-360, d['X'][0,:ncut])) #transformando para -180 a 180
    mask = concatenate((mask[:,ncut:], mask[:,:ncut]), axis=1)
    ilo = array([where(lon_m <= -63)[0].max(), where(lon_m >= 15)[0].min()]) #selecionando minha area de interesse
    lon_m = lon_m[ilo[0]+1:ilo[1]-1]
    mask = mask[:,ilo[0]+1:ilo[1]-1]
    #d['X'][:,:].shape
    # set basin names
    basin = ('atlantic', 'pacific', 'indian', 'caribe',
             'golfo', 'newze', 'bengal')
    # mask contains 14 extra latitudes near the poles
    mask = mask[7:-7, :]
    return mask, basin
# -----------------------------------------------------------------------------


def get_Rrad():
    # read 1st BC Rossby radii and gravity wave phase speed from OSU
    # site interpolated by grid_rossrad.py (contains nans and spurious
    # values under continents)
    d = load('dados_auxiliares/RossbyRadii_and_c.npz')
    Rr = d['Rr']  # in km
    c1 = d['c']  # in km/day
    # Rrg and c1 also contain 14 extra latitudes near the poles
    Rr = Rr[7:-7, :]
    c1 = c1[7:-7, :]
    Rr = ma.masked_where(isnan(Rr), Rr)
    c1 = ma.masked_where(isnan(c1), c1)
    return Rr, c1
# -----------------------------------------------------------------------------


def get_zg(lat, path):
    # read SSHA from reshaped cmems data generated by
    # xy2xt_xr_cmems_allyears.py. nf is a latitude index that goes from
    # 1 to ny nt and nx are obtained from get_aux() above
    sy0 = 1993
    sy1 = 2022  # dt.date.today().year
    wrf = 'alt_l4_c3s_climate_lat{0:05.3f}_{1:}_{2:}'.format(lat, sy0, sy1) + '.nc'
    fn = path + wrf
    print('[get_zg] Reading {:} ...\n'.format(fn))
    ds = Dataset(fn, 'r')
    # convert from m to mm
    zg = 1000.0*ds['sla'][:,:]
    zg = ma.masked_invalid(zg)
    zg.set_fill_value(0.0)
    return zg
# -----------------------------------------------------------------------------


def get_gparms(lat):
    Omega = 2*pi  # 1/day
    Re = 6380.0  # radius of the planet in km
    f0 = 2.*Omega*sin(d2r(lat))   # in 1/day
    beta = 2*Omega*cos(d2r(lat))/Re   # in 1/(km.day)
    return f0, beta, Omega, Re
