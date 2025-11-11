#!/home/ana/anaconda3/bin/python3
# *- coding: utf-8 -*-
"""
Created on Fry Sep 8 09:16 2023

@author: anamariani

Versão Modificada de fiter_z criado por @paulopolito para aplicação de filtro semi-automatico nos dados de 
altímetro da região do Atlântico Tropical, entre 6°N e 6°S
"""

def d2r(a):
	#convert degrees to radianos
    return pi*a/180

import numpy as np
from numpy.linalg import inv
import xarray as xr
from scipy.fft import fft, fftfreq, fftshift
from scipy.signal import find_peaks
from scipy.stats import mode
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from scipy import optimize as opt

def polifit(z, t):
	'''
	Fitting a linear exponential regression.
	t = time [numpy.array]
	z = monthly variable [numpy.array]
	'''
	N = len(t)
	um = np.ones(N)
	# build model function matrix for y = a + b*t + c*(t**2) + d*(t**3)
	A = np.column_stack((um, um*t, um*(t**2), um*(t**3)))
	# do the projection (A'A)\(A'z)
	a, b, c, d = np.matmul(inv(np.matmul(A.T, A)), np.matmul(A.T, z))
	# assemble fitted curve
	zfit = a + b*t + c*(t**2) + d*(t**3)

	return zfit


class FFT:
	def __init__ (self, z, delt, delx, lat):
		#Calcula a Transformada de Furrier em
		#cada longitude e em cada tempo de z
		(npt, npx) = z.shape
		fact = abs(fft(z, axis=0)) #Para encontrar os períodos predominantes
		facx = abs(fft(z, axis=1)) #Para encontrar os comprimentos de onda predominantes
		ft = fftfreq(fact.shape[0], d=delt)
		fx = fftfreq(facx.shape[1], d=delx*111.195*np.cos(d2r(lat)))
		#  use 1 + np* to avoid the DC -  reorganiza as transformadas de Fourier usando a função 'fftshift', movendo a componente DC (frequência zero) para o centro
		self.fact = fftshift(fact)[1 + npt//2:, :]
		self.facx = fftshift(facx)[:, 1 + npx//2:]
		#facx[:,:15] = detrend(facx[:,:15], axis=1)
		self.ft = fftshift(ft)[1 + npt//2:]
		self.fx = fftshift(fx)[1+npx//2:]
		tt = 1/(3*360)
		ni = np.where(ft >= tt)[0][0]
		nf = np.where(ft >= 1e-1)[0][0]
		self.ft = ft[ni:nf]
		self.fact = fact[ni:nf,:]

	def Exp(self, t, param):
		m, b, a = param
		return m * np.exp(-b*t) + a
		#return np.exp(-param[0] * t) * np.sum([np.power(t, i - 1) * param[i] for i in range(1, len(param))], axis=0)
		#return np.sum([np.power(t, i) * param[i] for i in range(len(param))], axis=0)

	def chi_sq(self, param, z, t):
		error = np.array([zval - self.Exp(tval, param) for zval, tval in zip(z, t)])
	#	newt = np.sum([np.power(t, i) * param[i] for i in range(len(param))], axis=0)
	#	error = np.array([zval - tval for zval, tval in zip(z, newt)])
		return np.dot(error, error)


	def expfit2(self,z, t):
		'''
		Fitting a non-linear exponential regression.
		t = time [numpy.array]
		z = monthly variable [numpy.array]
		'''
		#params, cv  = curve_fit(self.Exp, t, z, full_output=True)
		params = np.ones(3) * 100
		for i in range(4):
			params = opt.fmin_powell(self.chi_sq, params, args=(z, t,), full_output=True)
			params = params[0]
			print(params)
		#m, b, a = params
		zfit = self.Exp(t, params)
		
		return zfit

	def rm_exp(self, facf, ff):
		for i in range(facf.shape[1]):
			zfit_t = self.expfit2(facf[:,i], ff)
			facf[:,i] = facf[:,i] - zfit_t
		return facf

	def rm_exp_T(self):
		return self.rm_exp(self.fact, self.ft)

	def rm_exp_L(self):
		return self.rm_exp(self.facx.T, self.fx)


	def peacks(self, st, n):
		'''
		Encontra os n maiores picos de uma série temporal
		'''
		p,_ = find_peaks(st)
		picos_ordenados = sorted(p, key=lambda x: st[x], reverse=True)
		p4 = picos_ordenados[:n]
		return p4

	def find_major_peacks(self, facf, ff, n):
		'''
		Encontra os n maiores picos para cada uma das séries
		na matriz facf
		'''
		#Para o tempo
		xs = facf.shape[0]
		ys = facf.shape[1]
		f4 = np.ones((n, ys))*np.nan
		facf4 = np.ones((n, ys))*np.nan
		for i in range(ys):
			p4 = self.peacks(facf[:,i], n)
			f4[:len(p4),i] = ff[p4]
			facf4[:len(p4),i] = facf[p4,i]
		return f4, facf4

	def major_peacks_T(self, n):
		return self.find_major_peacks(self.fact, self.ft, n)

	def major_peacks_L(self, n):
		return self.find_major_peacks(self.facx.T, self.fx, n)

	def mode_peacks_T(self, n):
		'''
		Calcula a moda para cada um dos picos encontrados
		nas séries de fft ao longo do tempo
		'''
		f1,facf1 = self.major_peacks_T(n)
		mf1,_ = mode(f1, axis=1)
		mfacf1,_ = mode(facf1, axis=1)
		return mf1, mfacf1

	def mode_peacks_L(self, n):
		'''
		Calcula a moda para cada um dos picos encontrados
		nas séries de fft ao longo do espaço
		'''
		f1,facf1 = self.major_peacks_L(n)
		mf1,_ = mode(f1, axis=1)
		mfacf1,_ = mode(facf1, axis=1)
		return mf1, mfacf1

	def show_FFT_fit(self, n):

		plt.ion()
		fig, ax = plt.subplots(1, 2, num=5, figsize=(15, 8), clear=True)
		ax[0].scatter(self.ft, self.fact.mean(axis=1), color='gray', zorder=1) #.mean(axis=1)
		zfit_t = self.expfit2(self.fact[:,0], self.ft)
		#ax[0].semilogx(self.ft, self.fact.mean(axis=1)-zfit_t, color='gray', zorder=1)
		ax[0].plot(self.ft,zfit_t, color='k', ls='--', lw=0.8, zorder=2)
		ax[1].scatter(self.fx, self.facx.mean(axis=0), color='gray', zorder=1) #.mean(axis=0)
		zfit_x = self.expfit2(self.facx[0,:], self.fx)
		#ax[1].semilogx(self.fx, self.facx.mean(axis=0)-zfit_x, color='gray', zorder=1)
		ax[1].plot(self.fx,zfit_x, color='k', ls='--', lw=0.8, zorder=2)
		ax[0].set_xlabel('Frequências [1/dia]')
		ax[0].set_xlim(1e-4,0.1)
		ax[1].set_xlabel('Comprimentos de onda [1/km]')
		#ax2[1].set_xlim(0,0.005)
		ax[0].set_ylabel('Potências')

		return fig, ax

	def show_FFT_mean(self, n):

		plt.ion()
		fig, ax = plt.subplots(1, 2, num=10, figsize=(15, 8), clear=True)
		ax[0].loglog(self.ft, np.nanmean(self.fact,axis=1), color='gray', zorder=1) #.mean(axis=1)
		ax[1].loglog(self.fx, np.nanmean(self.facx, axis=0), color='gray', zorder=1) #.mean(axis=0)
		col = ['r', 'b', 'g', 'k', 'y'] #Paleta de cores vermelhas que vai indo para o rosa a medida que diminui
		pt = self.peacks(np.nanmean(self.fact,axis=1), n)
		ft4 = self.ft[pt]
		fact4 = np.nanmean(self.fact,axis=1)[pt]
		px = self.peacks(np.nanmean(self.facx,axis=0), n)
		fx4 = self.fx[px]
		facx4 = np.nanmean(self.facx,axis=0)[px]
		for i in range(n):
			ax[0].scatter(ft4[i], fact4[i], c=col[i], s=5, zorder=5, label='{:.0f} dias'.format(1/ft4[i]))
			if i == 0:
				ax[1].scatter(fx4[i], facx4[i], c=col[i], s=5, zorder=5, label='{:.0f} km'.format(1/fx4[i]))
		ax[0].set_xlabel('Frequências [1/dia]')
		ax[0].set_xlim(1e-3,0.1)
		ax[1].set_xlabel('Comprimentos de onda [1/km]')
		#ax2[1].set_xlim(0,0.005)
		ax[0].set_ylabel('Potências')
		ax[0].legend()
		ax[1].legend()
		#Text in the rigth side of the figure
		trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
		ax[0].text(0.0, 0.98, '(a)', zorder=100, size=10.0, transform=ax[0].transAxes + trans, verticalalignment='top', 
			bbox=dict(facecolor='0.8', edgecolor='k', pad=3.0))
		ax[1].text(0.1, 0.98, '(b)', zorder=100, size=10.0, transform=ax[1].transAxes + trans, verticalalignment='top', 
			bbox=dict(facecolor='0.8', edgecolor='k', pad=3.0))

		plt.pause(20)

		return fig, ax, ft4, fx4

	def show_FFT_todos(self, n):

		plt.ion()
		fig, ax = plt.subplots(1, 2, num=15, figsize=(15, 8), clear=True)
		ax[0].loglog(self.ft, self.fact, color='gray', lw=0.7, zorder=1) #.mean(axis=1)
		ax[1].loglog(self.fx, self.facx.T, color='gray', lw=0.7, zorder=1) #.mean(axis=0)
		col = ['r', 'b', 'g', 'k', 'y'] #Paleta de cores vermelhas que vai indo para o rosa a medida que diminui
		ft4,fact4 = self.major_peacks_T(n)
		fx4,facx4 = self.major_peacks_L(n)
		for i in range(n):
			ax[0].scatter(ft4[i,:], fact4[i,:], c=col[i], s=2, zorder=5, label='{}'.format(i))
			ax[1].scatter(fx4[i,:], facx4[i,:], c=col[i], s=2, zorder=5, label='{}'.format(i))
		ax[0].set_xlabel('Frequências [1/dia]')
		ax[0].set_xlim(1e-4,0.1)
		ax[1].set_xlabel('Comprimentos de onda [1/km]')
		#ax2[1].set_xlim(0,0.005)
		ax[0].set_ylabel('Potências')
		ax[0].legend()
		ax[1].legend()
		plt.pause(20)

		return fig, ax

	def show_FFT_sem_exp(self, n):

		plt.ion()
		fig, ax = plt.subplots(1, 2, num=20, figsize=(15, 8), clear=True)
		fact = self.rm_exp_T()
		facx = self.rm_exp_L()
		ax[0].plot(self.ft, fact, color='gray', lw=0.7, zorder=1) #.mean(axis=1)
		ax[1].plot(self.fx, facx, color='gray', lw=0.7, zorder=1) #.mean(axis=0)
		col = ['r', 'b', 'g', 'k', 'y'] #Paleta de cores vermelhas que vai indo para o rosa a medida que diminui
		ft4,fact4 = self.find_major_peacks(fact, self.ft, n)
		fx4,facx4 = self.find_major_peacks(facx, self.fx, n)
		for i in range(n):
			ax[0].scatter(ft4[i,:], fact4[i,:], c=col[i], s=2, zorder=5, label='{}'.format(i))
			ax[1].scatter(fx4[i,:], facx4[i,:], c=col[i], s=2, zorder=5, label='{}'.format(i))
		ax[0].set_xlabel('Frequências [1/dia]')
		ax[0].set_xlim(1e-4,0.1)
		ax[1].set_xlabel('Comprimentos de onda [1/km]')
		#ax2[1].set_xlim(0,0.005)
		ax[0].set_ylabel('Potências')
		ax[0].legend()
		ax[1].legend()
		
		return fig, ax

	def show_hist(self, n, cm):
		plt.ion()
		fig, ax = plt.subplots(1, 1, num=25, figsize=(8, 8), clear=True)
		fx4, facx4 = self.major_peacks_L(n)
		ax.hist(fx4[cm,:], bins=80)
		ax.set_title('Componente {}'.format(cm))
		ax.set_xlabel('Número de onda (K) [1/km]')
		plt.pause(20)

		return fig, ax






