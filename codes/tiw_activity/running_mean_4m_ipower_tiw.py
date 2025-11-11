def mov_mean(z, p):
	'''
	Calcula média móvel de
	z com p dias.
	Código de @polito
	'''
	n = len(z)
	s = np.empty(n)
	p2 = p//2
	for i in range(n):
		if (i - p2) < 0:
			i0 = 0
		else:
			i0 = i - p2
		if (i + (p - p2)) > n:
			i1 = n
		else:
			i1 = i + (p - p2)
		s[i] = np.nanmean(z[i0:i1])
	return s


import numpy as np
import matplotlib.pyplot as plt

f = np.load('/data/anamariani/resultados_mestrado/wavelets/mapa_tendencia_Pot_tiw_com_var_bootstrap_method_new.npz')

#f = np.load('/home/anamariani/Documents/Resultados/resultados_mestrado/wavelets/mapa_tendencia_Pot_tiw_total.npz')
#g = np.load('/home/anamariani/Documents/Resultados/resultados_mestrado/wavelets/desvio_padrao_quad_comp_30_dias.npz')

t = f['time']
#peaks=f['ipower']
#dv = g['dv']
ipower=f['ipower'] #potencia da onda integrada na frequencia após aplicar a transformada de ondaleta contendo todos os pontos

lat=f['lat']
lon=f['lon']

power = np.ones(ipower.shape)*np.nan

for i, la in enumerate(lat):
	for j, lo in enumerate(lon):
		#média móvel de 4 meses da série de potencia da onda
		#power[:,i,j] = mov_mean(ipower[:,i,j]*dv[i,j],120)
		power[:,i,j] = mov_mean(ipower[:,i,j],120)
	print('Latitude {:.3f} ({}/{}) ok!'.format(la, i, len(lat)))


np.savez('/data/anamariani/resultados_mestrado/wavelets/Pot_tiw_total_run_mean_new.npz',time=t,lat=lat,lon=lon, pot_mean=power)

