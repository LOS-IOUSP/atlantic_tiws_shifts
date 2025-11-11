#!/home/ana/anaconda3/bin/python3
# *- coding: utf-8 -*-
"""
Created on Mon Jun 10 16:41 2024

@author: anamariani

Classe modificada da class_filter para selecionar ondas em um Hovmoller ja filtrado
para identificar o periodo, comprimento de onda de cada pacote, além da posição dele
no hovmoller.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cmocean.cm as cmo
from scipy.interpolate import interp1d
import pandas as pd
from class_fft import FFT
from collections import OrderedDict as od



class select_waves:

	def __init__(self, z, delt, delx, lat,xn,tn):
		plt.ion()
		fig, ax = plt.subplots(1, 1, num=1, figsize=(8, 12), clear=True)
		vmin = np.nanmean(z) - 3*np.nanstd(z)
		vmax = np.nanmean(z) + 3*np.nanstd(z)

		plt.pcolormesh(xn, tn, z, vmin=vmin, vmax=vmax, cmap=cmo.curl, shading='auto')

		T0, Te0, L0, temp_m, lon_m, temps, lons = np.array([]), np.array([]), np.array([]),np.array([]),np.array([]),{},{}
		fft = FFT(z, delt, delx, lat)
		T2 = 1/fft.mode_peacks_T(1)[0][0][0]
		L2 = 1/fft.mode_peacks_L(1)[0][0][0]
		cp2 = L2/T2
		print('A análise por FFT sugere que você procure por ondas com',
				'T={:.0f} dias, L={:.0f} km, cp={:.1f} km/dia'.format(
					T2, L2, cp2))

		Ld2 = L2/(111.195*np.cos(lat*np.pi/180))
		# faz o grid com L e T
		ax.xaxis.set_ticks(np.round(np.arange(xn[0], xn[-1], Ld2)))
		ax.yaxis.set_ticks(np.round(np.arange(tn[0], tn[-1], T2)))
		ax.grid()
		loop_k = 0
		do_this_component = True
		while do_this_component:
			if loop_k == 0:
				print('\n------------------------------------------------------')
				print('Por favor aumente bem o tamanho da janela e amplie',
					' uma região com ondas bem claras.\n Você deve selecionar',
					' cristas e cavados adjacentes do que parecem ser',
					' as ondas de período mais longo.')
				print('Botão esquerdo seleciona, direito apaga e do meio termina')

			print('Selecione até 11 pontos com o mouse sobre uma CRISTA')
			xt0 = []
			while len(xt0) == 0:
				xt0 = np.array(plt.ginput(n=11, timeout=0))
			f0 = interp1d(xt0[:, 0], xt0[:, 1])
			ax.plot(xt0[:, 0], xt0[:, 1], color='g', marker='o', linestyle='none')
			print('Selecione até 11 pontos com o mouse sobre o próximo CAVADO')
			xt1 = []
			while len(xt1) == 0:
				xt1 = np.array(plt.ginput(n=11, timeout=0))
			ax.plot(xt1[:, 0], xt1[:, 1], color='r', marker='s', linestyle='none')
			plt.show()
			f1 = interp1d(xt1[:, 0], xt1[:, 1])
			# find common range between xt0 and xt1 longitudes
			xr0 = np.max(np.array([xt0[:, 0].min(), xt1[:, 0].min()]))
			xr1 = np.min(np.array([xt0[:, 0].max(), xt1[:, 0].max()]))
			# build regular lon grid rounded to the next 0.25
			x0 = np.arange(np.ceil(xr0*4)/4, np.floor(xr1*4)/4, 0.25)
			# these are the regularly gridded mouse inputs
			t0 = f0(x0)
			t1 = f1(x0)
			# save the phase speed in km/day for this crest/through pair
			cp_crista = np.mean((np.diff(x0)/np.diff(t0))*111.195*np.cos(lat*np.pi/360))
			cp_cavado = np.mean((np.diff(x0)/np.diff(t1))*111.195*np.cos(lat*np.pi/360))
			cpe_crista = np.std((np.diff(x0)/np.diff(t0))*111.195*np.cos(lat*np.pi/360))
			cpe_cavado = np.std((np.diff(x0)/np.diff(t1))*111.195*np.cos(lat*np.pi/360))
			if loop_k == 0:
				cp0 = np.array([cp_crista, cp_cavado])
				cpe0 = np.array([cpe_crista, cpe_cavado])
			else:
				cp0 = np.vstack((cp0, np.array([cp_crista, cp_cavado])))
				cpe0 = np.vstack((cp0, np.array([cpe_crista, cpe_cavado])))
			# save Period in days
			T0 = np.append(T0, 2*np.mean(np.abs(t0-t1)))
			Te0 = np.append(Te0, 2*np.std(np.abs(t0-t1)))
			#Save the Wave lengh in km:
			wlen = ((cp_crista + cp_cavado)/2)*(2*np.mean(np.abs(t0-t1)))
			L0 = np.append(L0, wlen)
			#save the position of the waves:
			temps[loop_k] = np.array([t0,t1]) #crista e cavado
			lons[loop_k] = x0
			#save the mean position of the waves:
			temp_m = np.append(temp_m,(np.mean(t0)+np.mean(t1))/2)
			lon_m = np.append(lon_m, np.mean(x0))
			# we actually dont need to save the wavelength if we have cp and T
			plt.plot(x0, t0, color='k')
			#print(x0)
			#print(t0)
			plt.plot(x0, t1, color='k', linestyle='--')
			#print(t1)
			sn = input('Quer selecionar outro par crista & cavado? [s/n]\n')
			loop_k += 1
			if sn == 'n':
				do_this_component = False
		# cp1, L1 e T1 são nossas estimativas manuais de velocidade de fase
		# comprimento e período que serão usadas para construir o filtro
		self.T0 = T0
		self.L0 = L0
		self.cp0_npz = cp0
		self.cp0 = cp0.ravel()
		self.T1 = T0.mean()
		self.Te1 = Te0.mean()/np.sqrt(np.array(len(T0)))
		self.cp1 = cp0.mean()
		self.cpe1 = cpe0.mean()/np.sqrt(np.array(len(cp0)))
		self.L1 = self.cp1*self.T1
		self.Le1 = self.L1*(self.Te1/self.T1 + self.cpe1/self.cp1)
		self.z = z
		self.delt = delt
		self.delx = delx
		self.lat = lat
		self.xn = xn
		self.tn = tn
		self.temps = temps
		self.lons = lons
		self.temp_m = temp_m
		self.lon_m = lon_m
		print('[manual_filter] --- supervised parameters ---')
		print('[manual_filter] T1=({:.1f}+-{:.1f}) days'.format(self.T1, self.Te1))
		print('[manual_filter] cp1=({:.2f}+-{:.2f}) km/day'.format(self.cp1, self.cpe1))
		print('[manual_filter] L1=({:.0f}+-{:.0f}) km\n'.format(self.L1, self.Le1))

	def get_param(self):
		A = np.sqrt(2)*np.nanmean(self.z)
		(npt, npx) = self.z.shape
		#analise de FFT
		fft = FFT(self.z, self.delt, self.delx, self.lat)
		#fig, ax, ft4, fx4 = fft.show_FFT_mean(2)
		#print(1/ft4)
		#print(1/fx4)
		#Find major_peacks
		mt, mfact = fft.major_peacks_T(1)
		mx, mfacx = fft.major_peacks_L(1)

		#Define os períodos e comprimentos de onda
		Tds = np.array(1/mt[0])
		Lkms = np.array(1/mx[0])

		gT = (Tds >= Tds.mean() - Tds.std()) &\
			(Tds <= Tds.mean() + Tds.std())
		T_d = np.mean(Tds[gT])
		T_de = np.std(Tds[gT])

		gL = (Lkms >= np.nanmean(Lkms) - np.nanstd(Lkms)) &\
			(Lkms <= np.nanmean(Lkms) + np.nanstd(Lkms))
		L_km = np.nanmean(Lkms[gL])
		L_kme = np.nanstd(Lkms[gL])

		cp_kmd = L_km/T_d
		cp_kmde = cp_kmd*(T_de/T_d + L_kme/L_km)

		print(np.nanmean(Lkms))
		print(Lkms)

		return cp_kmd, cp_kmde, L_km, L_kme, T_d, T_de, A

	def df_selected(self):
		cp_fft, _, L_fft, L_ffte, T_fft, T_ffte, _ = self.get_param()
		d = {}
		d = {'T selected':self.T0, 'Cp selected': self.cp0, 'T filter': np.ones(self.T0.shape)*self.T1, 
			'L filter': np.ones(self.T0.shape)*self.L1, 'T fft': np.ones(self.T0.shape)*T_fft, 
			'Te fft':np.ones(self.T0.shape)*T_ffte, 'L fft': np.ones(self.T0.shape)*L_fft, 
			'Le fft': np.ones(self.T0.shape)*L_ffte}
		df = pd.DataFrame.from_dict(d, orient='index').T

		print('[manual_filter] --- unsupervised parameters ---\n',
			'[manual_filter] Equatorial wave component ',
			'[manual_filter] found, T0 = {:.0f} days,\n'.format(T_fft),
			'[manual_filter] L0={:.0f} km, and cp={:.1f} km/d.'.format(L_fft, cp_fft))

		return df

	def save_param(self,file_name):
		np.savez(file_name,z=self.z, lat=self.lat, lon=self.xn, time=self.tn, T=self.T0,Cp=self.cp0_npz,L=self.L0, 
			mean_temps=self.temp_m, mean_lons=self.lon_m, temps=self.temps, lons=self.lons)
		return self.T0, self.L0, self.cp0_npz, self.temp_m, self.lon_m, self.temps, self.lons

	def retn_cp(self):
		return self.cp1, self.T1