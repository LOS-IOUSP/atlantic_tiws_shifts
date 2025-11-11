def eq_time_series(x,y,m):
	'''
	Função recursiva que iguala o número de
	pontos de duas arrays, adicionando pontos
	à série que tem menos pontos. O valor do 
	ponto adicionado é dado pela mediana dos pontos.
	X: primeira série [np array]
	Y: segunda série [np array]
	m: valor da mediana dos pontos [float]
	'''
	#Confere se séries tem o mesmo número de pontos
	if len(x) != len(y):
		#se não tiverem, ver qual a menor
		vmin = min(len(x),len(y))
		#adicionar um ponto na menor com o valor da mediana
		if len(x) == vmin:
			diff = len(y) - len(x)
			print(diff)
			x = np.concatenate((x,np.ones(diff)*m))
		else:
			diff = len(x) - len(y)
			print(diff)
			y = np.concatenate((y,np.ones(diff)*m))
		#Repetir o processo
		return eq_time_series(x,y,m)
	else:
		#se forem iguais, retorna as séries
		return x,y




import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
from statistics import median
from scipy.stats import wilcoxon, ranksums


rc('font', family={'family': 'serif', 'serif': ['Times']})
rc('font', size=16)
rc('text', usetex=True)

f = np.load('/data/anamariani/resultados_mestrado/wavelets/mapa_tendencia_Pot_tiw.npz')
#f = np.load('/home/anamariani/Documents/Resultados/resultados_mestrado/wavelets/mapa_tendencia_Pot_tiw_inov_and_wilcox_95percent_method_all_points.npz')
g = np.load('/data/anamariani/resultados_mestrado/wavelets/desvio_padrao_quad_comp_30_dias.npz')

t = f['time']
#peaks=f['ipower']
dv = g['dv']
peaks11=f['peaks']

lat=f['lat']
lon=f['lon']

peaks = np.ones(peaks11.shape)*np.nan

for i, la in enumerate(lat):
	for j, lo in enumerate(lon):
		peaks[:,i,j] = peaks11[:,i,j]*dv[i,j]
	


npeaks= peaks[~np.isnan(peaks)].ravel()


n = len(t)//2
peaks1 = peaks[:n,:,:]
n1peaks= peaks1[~np.isnan(peaks1)].ravel()
peaks2 = peaks[n:,:,:]
n2peaks= peaks2[~np.isnan(peaks2)].ravel()

#n1peaks, n2peaks = eq_time_series(n1peaks, n2peaks,median(npeaks))

print('Todos:')
print(ranksums(n1peaks,n2peaks))

plt.ion()
#plt.figure()
#sns.histplot(npeaks,color="dodgerblue", bins=100, kde=True)
#plt.ylabel('Frequência')
#plt.xlabel('Potência (W) [mm²]')
#plt.title('Distribuição dos Picos de Potência (TODOS)')

#fig, ax = plt.subplots(1,2, figsize=(10, 4), sharey=True)
#sns.histplot(n1peaks,color="deeppink", ax=ax[0], bins=100, kde=True)
#ax[0].set_ylabel('Frequência')
#ax[0].set_xlabel('Potência (W) [mm²]')
#ax[0].set_title('Picos Primeira metade (TODOS)')

#sns.histplot(n2peaks,color="limegreen", ax=ax[1], bins=100, kde=True)
#ax[1].set_ylabel('Frequência')
#ax[1].set_xlabel('Potência (W) [mm²]')
#ax[1].set_title('Picos Segunda metade (TODOS)')

print('Fazendo primeira figura')
fig = plt.figure()
sns.histplot(n2peaks,color="deeppink", label='Segunda Metade', bins=100, kde=True)
sns.histplot(n1peaks,color="limegreen", label='Primeira Metade', bins=100, kde=True)
plt.ylabel('Frequência')
plt.xlabel('Potência (W) [mm²]')
plt.title('Distribuição dos Picos de Potência (TODOS)')
plt.legend()
#fig.savefig('/home/anamariani/Documents/Resultados/figuras_teste/wavelets/Distribuicao_todas_localizacoes_ipower.png',bbox_inches='tight', dpi=300)
fig.savefig('/home/anamariani/Documents/Resultados/figuras_teste/wavelets/Distribuicao_todas_localizacoes_peaks_vezes_variancia.png',bbox_inches='tight', dpi=300)
print('Figura salva')


n = len(t)//2
max_lat = np.where(lat==6.125)[0][0]
min_lat = np.where(lat==1.625)[0][0]
max_lon = np.where(lon==-10.125)[0][0]
min_lon = np.where(lon==-40.125)[0][0]
peaks1 = peaks[:n,min_lat:max_lat,min_lon:max_lon]
n1peaks= peaks1[~np.isnan(peaks1)].ravel()
peaks2 = peaks[n:,min_lat:max_lat,min_lon:max_lon]
n2peaks= peaks2[~np.isnan(peaks2)].ravel()

print('HN:')
print(ranksums(n1peaks,n2peaks))

print(n1peaks.max())
print(n2peaks.max())

#n1peaks, n2peaks = eq_time_series(n1peaks, n2peaks,median(npeaks))

#plt.ion()
#plt.figure()
#sns.histplot(npeaks,color="dodgerblue", bins=100, kde=True)
#plt.ylabel('Frequência')
#plt.xlabel('Potência (W) [mm²]')
#plt.title('Distribuição dos Picos de Potência (TODOS - HN)')

#fig, ax = plt.subplots(1,2, figsize=(10, 4), sharey=True)
#sns.histplot(n1peaks,color="deeppink", ax=ax[0], bins=100, kde=True)
#ax[0].set_ylabel('Frequência')
#ax[0].set_xlabel('Potência (W) [mm²]')
#ax[0].set_title('Picos Primeira metade (TODOS - HN)')

#sns.histplot(n2peaks,color="limegreen", ax=ax[1], bins=100, kde=True)
#ax[1].set_ylabel('Frequência')
#ax[1].set_xlabel('Potência (W) [mm²]')
#ax[1].set_title('Picos Segunda metade (TODOS - HN)')

print('Fazendo segunda figura')
fig2 = plt.figure()
sns.histplot(n2peaks,color="deeppink", label='Second half', bins=100, kde=True)
sns.histplot(n1peaks,color="limegreen", label='First half', bins=100, kde=True)
plt.ylabel('Number of Peaks')
plt.xlabel('Power [mm²]')
#plt.title('Distribuição dos Picos de Potência (TODOS - HN)')
plt.xlim([0.0079,0.35])
plt.legend()
#fig2.savefig('/home/anamariani/Documents/Resultados/figuras_teste/wavelets/Distribuicao_HN_ipower.png',bbox_inches='tight', dpi=300)
fig2.savefig('/home/anamariani/Documents/Resultados/figuras_teste/wavelets/Distribuicao_HN_restrito_peaks_vezes_variancia_english.png',bbox_inches='tight', dpi=300)
print('Figura salva')


n = len(t)//2
max_lat = np.where(lat==-0.125)[0][0]
min_lat = np.where(lat==-7.625)[0][0]
max_lon = np.where(lon==-0.125)[0][0]
min_lon = np.where(lon==-30.125)[0][0]
peaks1 = peaks[:n,min_lat:max_lat,min_lon:max_lon]
n1peaks= peaks1[~np.isnan(peaks1)].ravel()
peaks2 = peaks[n:,min_lat:max_lat,min_lon:max_lon]
n2peaks= peaks2[~np.isnan(peaks2)].ravel()

#n1peaks, n2peaks = eq_time_series(n1peaks, n2peaks,median(npeaks))

#plt.ion()
#plt.figure()
#sns.histplot(npeaks,color="dodgerblue", bins=100, kde=True)
#plt.ylabel('Frequência')
#plt.xlabel('Potência (W) [mm²]')
#plt.title('Distribuição dos Picos de Potência (TODOS - HS)')

#fig, ax = plt.subplots(1,2, figsize=(10, 4), sharey=True)
#sns.histplot(n1peaks,color="deeppink", ax=ax[0], bins=100, kde=True)
#ax[0].set_ylabel('Frequência')
#ax[0].set_xlabel('Potência (W) [mm²]')
#ax[0].set_title('Picos Primeira metade (TODOS - HS)')

#sns.histplot(n2peaks,color="limegreen", ax=ax[1], bins=100, kde=True)
#ax[1].set_ylabel('Frequência')
#ax[1].set_xlabel('Potência (W) [mm²]')
#ax[1].set_title('Picos Segunda metade (TODOS - HS)')

print('HS:')
print(ranksums(n1peaks,n2peaks))

print(n1peaks.max())
print(n2peaks.max())

print('Fazendo terceira figura')
fig3 = plt.figure()
sns.histplot(n2peaks,color="deeppink", label='Second half', bins=100, kde=True)
sns.histplot(n1peaks,color="limegreen", label='First half', bins=100, kde=True)
plt.ylabel('Number of Peaks')
plt.xlabel('Power [mm²]')
#plt.title('Distribuição dos Picos de Potência (TODOS - HS)')
plt.xlim([0.0049,0.14])
plt.legend()
fig3.savefig('/home/anamariani/Documents/Resultados/figuras_teste/wavelets/Distribuicao_HS_peaks_restrito_vezes_variancia_english.png',bbox_inches='tight', dpi=300)
#fig2.savefig('/home/anamariani/Documents/Resultados/figuras_teste/wavelets/Distribuicao_HN_peaks_vezes_variancia.png',bbox_inches='tight', dpi=300)
print('Figura salva')
