import pandas as pd
import numpy as np
import GPy, scipy
from patsy.contrasts import Helmert

class GP_FANOVA(object):

	EFFECT_SUFFIXES = ['alpha','beta','gamma','delta','epsilon']

	def __init__(self,x,y,effect):
		""" Base model for the GP FANOVA framework.

		Input must be of the form:
		x: n x p
		y: n x r
		effect: r x k

		where n is the number of sample points, p is the input dimension,
		r is the number of replicates, and k is the number of effects.
		"""

		# store indexers
		self.x = x # independent variables
		self.y = y # dependent variables
		self.effect = effect # effect variables

		self.n = self.x.shape[0]
		assert self.y.shape[0] == self.n, 'x and y must have same first dimension shape!'
		self.p = self.x.shape[1]

		self.r = self.nt = self.y.shape[1]
		assert self.r == self.effect.shape[0], 'y second dimension must match effect first dimension'

		self.k = self.effect.shape[1] # number of effects
		self.mk = [np.unique(self.effect[:,i]).shape[0] for i in range(self.k)] # number of levels for each effect
		#self.nk = dict([(i,[sum(self.effect[:,i] == j) for j in range(self.mk[i])]) for i in range(self.k)]) # number of observations for each effect level

		ind = self.build_index()

		# print self.sample_n*(1+sum(self.mk))+4+self.k*2, len(ind)
		self.parameter_cache = pd.Series(np.zeros(len(ind)),
											index=ind)

		# set some initial values
		self.parameter_cache[['mu_sigma','mu_lengthscale','y_sigma','y_lengthscale']] = 1
		for i in range(self.k):
			self.parameter_cache[['%s*_sigma'%GP_FANOVA.EFFECT_SUFFIXES[i],'%s*_lengthscale'%GP_FANOVA.EFFECT_SUFFIXES[i]]] = 1
			for j in range(i):
				self.parameter_cache[['(%s:%s)*_sigma'%(GP_FANOVA.EFFECT_SUFFIXES[j],GP_FANOVA.EFFECT_SUFFIXES[i]),'(%s:%s)*_lengthscale'%(GP_FANOVA.EFFECT_SUFFIXES[j],GP_FANOVA.EFFECT_SUFFIXES[i])]] = 1
		self.parameter_cache['y_sigma'] = .1

		self.parameter_history = pd.DataFrame(columns=ind)

		# contrasts
		self.contrasts = [Helmert().code_without_intercept(range(self.mk[i])).matrix for i in range(self.k)]

	def build_index(self):
		# build the index for the parameter cache
		ind = self.mu_index() + ['mu_sigma','mu_lengthscale']
		for i in range(self.k):
			for j in range(self.mk[i]-1):
				ind += self.effect_contrast_index(i,j)
			ind += ["%s*_sigma"%GP_FANOVA.EFFECT_SUFFIXES[i],"%s*_lengthscale"%GP_FANOVA.EFFECT_SUFFIXES[i]]

			# add interaction samples
			for k in range(i):
				for l in range(self.mk[i]-1):
					for m in range(self.mk[k])-1:
						ind += self.effect_interaction_contrast_index(i,l,k,m)
				ind += ["(%s:%s)*_sigma"%(GP_FANOVA.EFFECT_SUFFIXES[k],GP_FANOVA.EFFECT_SUFFIXES[i]),
						"(%s:%s)*_lengthscale"%(GP_FANOVA.EFFECT_SUFFIXES[k],GP_FANOVA.EFFECT_SUFFIXES[i])]

		ind += ['y_sigma','y_lengthscale']

		return ind

	def offset(self):
		"""offset for the calculation of covariance matrices inverse"""
		return 1e-9

	def sample_prior(self,update_data=False):

		# this is broken without specifying self.n/points to sample from

		# mean
		mu,cov = np.zeros(self.n), self.mu_k().K(self.x) + np.eye(self.n)*self.offset()
		mean = scipy.stats.multivariate_normal.rvs(mu,cov)

		# effects
		effects = []
		for i in range(self.k):
			effects.append(np.zeros((self.n,self.mk[i])))
			for j in range(self.mk[i]):
				if j == self.mk[i] - 1:
					sample = -np.sum(effects[i],1)
					effects[i][:,j] = sample
				mu = np.zeros(self.n)
				mu -= np.sum(effects[i][:,:j],1) / (self.mk[i] - j)
				cov = 1.*(self.mk[i]-j-1)/(self.mk[i]-j) * self.effect_k(i).K(self.x) + np.eye(self.n)*self.offset()
				sample = scipy.stats.multivariate_normal.rvs(mu,cov)
				effects[i][:,j] = sample

		# effect interactions
		# cheating right now assuming exactly two effects
		effect_interactions = np.zeros(tuple([self.n]+self.mk))
		for i in range(self.mk[0]):
			for j in range(self.mk[1]):
				if i == self.mk[0]-1:
					effect_interactions[:,i,j] = -np.sum(effect_interactions[:,:i,j],1)
					continue
				if j == self.mk[1]-1:
					effect_interactions[:,i,j] = -np.sum(effect_interactions[:,i,:j],1)
					continue
				mu = np.zeros(self.n)
				mu -= np.sum(effect_interactions[:,:i,j],1)/(self.mk[0]-i)
				mu -= np.sum(effect_interactions[:,i,:j],1)/(self.mk[1]-j)
				cov = 1.*(self.mk[0]-i-1)/(self.mk[0]-i)*(self.mk[1]-j-1)/(self.mk[0]-j) * self.effect_interaction_k(0,1).K(self.x) + np.eye(self.n)*self.offset()
				effect_interactions[:,i,j] = scipy.stats.multivariate_normal.rvs(mu,cov)

		# noise
		obs = np.zeros((self.n,self.nt))
		for i in range(self.nt):
			mu = mean.copy()
			for j in range(self.k):
				mu+=effects[j][:,self.effect[i,j]]
				for k in range(j):
					mu+=effect_interactions[:,j,k]
			cov = self.y_k().K(self.x)
			obs[:,i] = scipy.stats.multivariate_normal.rvs(mu,cov)

		if update_data:
			self.y = obs

		return mean,effects,effect_interactions,obs

	def effect_contrast_index(self,k,l,):
		"""lth sample of kth effect"""
		return ['%s*_%d(%lf)'%(GP_FANOVA.EFFECT_SUFFIXES[k],l,z) for z in self.x]

	def effect_interaction_contrast_index(self,k,l,m,n):
		if k < m:
			return effect_interaction_contrast_index(m,n,k,l)

		return ['(%s_%d:%s_%d)*(%lf)'%(GP_FANOVA.EFFECT_SUFFIXES[k],l,GP_FANOVA.EFFECT_SUFFIXES[m],n,z) for z in self.x]

	def mu_index(self):
		return ['mu(%lf)'%z for z in self.x]

	def effect_contrast_array(self,i,k=None):
		a = np.zeros((self.n,self.mk[i]-1))
		for j in range(self.mk[i]-1):
			a[:,j] = self.parameter_cache[self.effect_contrast_index(i,j)]
		return a

	def effect_contrast_conditional_params(self,i,j):
		m = np.zeros(self.n)
		obs = np.zeros(self.n) # number of observations at each timepoint

		for k in range(self.mk[i]):
			if self.contrasts[i][k,j] == 0:
				continue

			reps = np.where(self.effect[:,i]==k)[0]
			temp = []
			for r in reps:
				temp2 = np.zeros(self.n)
				for l in range(self.mk[i]-1):
					if l == j:
						continue
					temp2 += self.contrasts[i][k,l] * self.parameter_cache[self.effect_contrast_index(i,l)]
				temp.append(self.y[:,r] - self.parameter_cache[self.mu_index()] - temp2)

				obs += 1 # all timepoints observed, need to update for nan's
			temp = np.array(temp)/self.contrasts[i][k,j]
			m += temp.mean(0)
		m /= self.mk[i]

		obs_cov_inv = np.linalg.inv(self.y_k().K(self.x)*obs)

		A = obs_cov_inv + np.linalg.inv(self.effect_contrast_k(i).K(self.x) + np.eye(self.n)*self.offset())
		b = np.dot(obs_cov_inv,m)

		A_inv = np.linalg.inv(A)
		return np.dot(A_inv,b), A_inv

	def y_sub_effects(self):

		yse = self.y.copy()
		for i in range(self.k):
			yse = yse - np.column_stack([self.parameter_cache[self.effect_index(i,k)] for k in self.effect[:,i]])

			for j in range(i):
				yse = yse - np.column_stack([self.parameter_cache[self.effect_interaction_index(i,k,j,l)] for k,l in zip(self.effect[:,i],self.effect[:,j])])

		return np.mean(yse,1)

	def y_sub_mu(self,i):
		ysm = self.y[:,self.effect==i] - self.parameter_cache[self.mu_index()].values[:,None]
		return np.sum(ysm,1)/self.nk[i]

	def y_sub_mu_for_effect(self,i,j):
		ysm = self.y[:,self.effect[:,i]==j] - self.parameter_cache[self.mu_index()].values[:,None]

		# remove interaction effects
		for k in range(self.k):
			if k == i:
				continue
			for l in range(self.mk[k]):
				if i > k:
					ysm = ysm - self.parameter_cache[self.effect_interaction_index(i,j,k,l)].values[:,None]
				else:
					ysm = ysm - self.parameter_cache[self.effect_interaction_index(k,l,i,j)].values[:,None]

		# return np.sum(ysm,1)/self.nk[i]
		return np.mean(ysm,1)

	def y_sub_mu_for_interaction(self,i,j,k,l):

		ysm = self.y[:,np.all((self.effect[:,i]==j,self.effect[:,k]==l),0)] - self.parameter_cache[self.mu_index()].values[:,None]

		ysm = ysm - self.parameter_cache[self.effect_index(i,j)].values[:,None] - self.parameter_cache[self.effect_index(k,l)].values[:,None]

		return ysm.mean(1)

	def mu_k(self):
		sigma,ls = self.parameter_cache[['mu_sigma','mu_lengthscale']]
		return GPy.kern.RBF(self.p,variance=sigma,lengthscale=ls)

	def effect_contrast_k(self,i):
		sigma,ls = self.parameter_cache[["%s*_sigma"%GP_FANOVA.EFFECT_SUFFIXES[i],"%s*_lengthscale"%GP_FANOVA.EFFECT_SUFFIXES[i]]]
		return GPy.kern.RBF(self.p,variance=sigma,lengthscale=ls)

	def effect_interaction_k(self,i,j):
		sigma,ls = self.parameter_cache[["%s:%s_sigma"%(GP_FANOVA.EFFECT_SUFFIXES[i],GP_FANOVA.EFFECT_SUFFIXES[j]),"%s:%s_lengthscale"%(GP_FANOVA.EFFECT_SUFFIXES[i],GP_FANOVA.EFFECT_SUFFIXES[j])]]
		return GPy.kern.RBF(self.p,variance=sigma,lengthscale=ls)

	def alpha_k(self):
		sigma,ls = self.parameter_cache[['alpha_sigma','alpha_lengthscale']]
		return GPy.kern.RBF(self.p,variance=sigma,lengthscale=ls)

	def y_k(self):
		sigma,ls = self.parameter_cache[['y_sigma','y_lengthscale']]
		# return GPy.kern.RBF(self.p,variance=sigma,lengthscale=ls)
		return GPy.kern.White(self.p,variance=sigma)

	def mu_conditional(self):
		offset = np.eye(self.sample_x.shape[0])*1e-9
		A = np.linalg.inv(self.mu_k().K(self.sample_x) + offset) + self.nt * np.linalg.inv(self.y_k().K(self.sample_x))
		# b = self.nt*np.dot(np.linalg.inv(self.y_k().K(self.sample_x)),self.y_sub_alpha())
		b = self.nt*np.dot(np.linalg.inv(self.y_k().K(self.sample_x)),self.y_sub_effects())

		A_inv = np.linalg.inv(A)
		return np.dot(A_inv,b), A_inv

	def effect_conditional(self,i,j):
		mu = np.zeros(self.sample_n)
		for k in range(j):
			mu -= self.parameter_cache[self.effect_index(i,k)]
		mu /= (self.mk[i] - j)

		if j == self.mk[i] - 1: # we're fuckin dun
			return mu,np.zeros((self.sample_n,self.sample_n))

		cov = 1.*(self.mk[i]-j-1)/(self.mk[i]-j) * self.effect_k(i).K(self.sample_x) + np.eye(self.sample_n)*self.offset()

		y_k_inv = np.linalg.inv(self.y_k().K(self.sample_x))
		k_effect_inv = np.linalg.inv(1.*(self.mk[i]-j-1)/(self.mk[i] - j) * self.effect_k(i).K(self.sample_x) + np.eye(self.sample_x.shape[0])*1e-9)

		A = k_effect_inv + self.nk[i][j] * y_k_inv
		b = self.nk[i][j]*np.dot(y_k_inv,self.y_sub_mu_for_effect(i,j)) + np.dot(k_effect_inv,mu)

		A_inv = np.linalg.inv(A)
		return np.dot(A_inv,b), A_inv

	def interaction_conditional(self,i,j,k,l):
		sub_effect_1,sub_effect_2 = np.zeros(self.sample_n), np.zeros(self.sample_n)

		# if i < k: # swapperino
		# 	temp1,temp2 = i,j
		# 	i,j = k,l
		# 	k,l = temp1,temp2

		for m in range(j):
			sub_effect_1 -= self.parameter_cache[self.effect_interaction_index(i,m,k,l)]
		for n in range(l):
			sub_effect_2 -= self.parameter_cache[self.effect_interaction_index(i,j,k,n)]

		# print sub_effect_1, sub_effect_2

		if j == self.mk[i] - 1:
			return sub_effect_1,np.zeros((self.sample_n,self.sample_n))
		elif l == self.mk[k] - 1:
			return sub_effect_2,np.zeros((self.sample_n,self.sample_n))
		else:
			mu = - sub_effect_1 - sub_effect_2

		mu -= mu/(self.mk[i]-j)/(self.mk[k]-l)
		cov = 1.*(self.mk[i]-j-1)/(self.mk[i]-j)*(self.mk[k]-l-1)/(self.mk[k]-l) * self.effect_interaction_k(i,k).K(self.sample_x) + np.eye(self.sample_n)*self.offset()

		# print mu,cov

		y_k_inv = np.linalg.inv(self.y_k().K(self.sample_x))

		A = cov + self.nk[i][j] * self.nk[k][l] * y_k_inv
		b = self.nk[i][j]*self.nk[k][l]*np.dot(y_k_inv,self.y_sub_mu_for_interaction(i,j,k,l)) + np.dot(cov,mu)

		A_inv = np.linalg.inv(A)
		return np.dot(A_inv,b), A_inv

	def alpha_conditional(self,i,order,params=False):
		if i == order[-1]: # enforce sum to zero constraint
			mu_alpha = np.zeros(self.sample_n)
			for j in order[:-1]:
				mu_alpha = mu_alpha - self.parameter_cache[self.alpha_index(j)].values
			return mu_alpha,np.zeros((self.sample_n,self.sample_n))

		ind = order.tolist().index(i)

		y_k_inv = np.linalg.inv(self.y_k().K(self.sample_x))
		# k_alpha_inv = np.linalg.inv(1.*(self.k-ind-1)/(self.k-ind) * self.alpha_k().K(self.sample_x))
		k_alpha_inv = np.linalg.inv(1.*(self.k-ind-1)/(self.k-ind) * self.alpha_k().K(self.sample_x) + np.eye(self.sample_x.shape[0])*1e-9)
		# k_alpha_inv = np.linalg.pinv(1.*(self.k-ind-1)/(self.k-ind) * self.alpha_k().K(self.sample_x))

		A = k_alpha_inv + self.nk[i] * y_k_inv

		mu_alpha = np.zeros(self.sample_n)
		for j in order[:ind]:
			mu_alpha = mu_alpha - self.parameter_cache[self.alpha_index(j)].values
		mu_alpha = mu_alpha / (self.k - ind)

		b = self.nk[i]*np.dot(y_k_inv,self.y_sub_mu(i)) + np.dot(k_alpha_inv,mu_alpha)
		# b = np.dot(k_alpha_inv,mu_alpha)
		# b = self.nk[i]*np.dot(y_k_inv,self.y_sub_mu(i))

		if params:
			return A,b

		A_inv = np.linalg.inv(A)
		return np.dot(A_inv,b), A_inv

	def update_mu(self):

		mu,cov = self.mu_conditional()
		sample = scipy.stats.multivariate_normal.rvs(mu,cov)
		self.parameter_cache.loc[self.mu_index()] = sample

	def update(self):

		# new_params = pd.DataFrame(self.parameter_history.iloc[-1,:]).T
		# new_params.index = [self.parameter_history.shape[0]]
		#
		# self.parameter_history = self.parameter_history.append(new_params)

		# update mu
		mu,cov = self.mu_conditional()
		sample = scipy.stats.multivariate_normal.rvs(mu,cov)
		self.parameter_cache.loc[self.mu_index()] = sample
		# print self.parameter_cache.loc[self.mu_index()]

		# update alpha
		# order = np.random.choice(range(self.k),self.k,replace=False)
		# for i in order:
		# 	mu,cov = self.alpha_conditional(i,order)
		# 	sample = scipy.stats.multivariate_normal.rvs(mu,cov)
		# 	self.parameter_cache.loc[self.alpha_index(i)] = sample

		# update dem effex
		for i in range(self.k):
			for j in range(self.mk[i]):
				mu,cov = self.effect_conditional(i,j)
				sample = scipy.stats.multivariate_normal.rvs(mu,cov)
				self.parameter_cache.loc[self.effect_index(i,j)] = sample

		# update dem interacshuns
		for i in range(self.k):
			for j in range(self.mk[i]):
				for k in range(i+1,self.k):
					for l in range(self.mk[k]):
						# print i,j,k,l
						mu,cov = self.interaction_conditional(i,j,k,l)
						sample = scipy.stats.multivariate_normal.rvs(mu,cov)
						self.parameter_cache.loc[self.effect_interaction_index(i,j,k,l)] = sample

		# update hyperparams

	def store(self):
		self.parameter_history = self.parameter_history.append(self.parameter_cache,ignore_index=True)
		self.parameter_history.index = range(self.parameter_history.shape[0])

	def sample(self,n=1,save=0):
		start = self.parameter_history.shape[0]
		i = 1
		while self.parameter_history.shape[0] - start < n:
			self.update()

			if save == 0 or i % save == 0:
				self.store()

			i+=1

	def plot_functions(self,plot_mean=True,offset=True,burnin=0):
		import matplotlib.pyplot as plt

		# plt.gca().set_color_cycle(None)
		colors = [u'b', u'g', u'r', u'c', u'm', u'y',]
		cmaps = ["Blues",'Greens','Reds']

		for i in range(self.k):
			if offset:
				mean = (self.parameter_history[self.mu_index()].values[burnin:,:] + self.parameter_history[self.alpha_index(i)].values[burnin:,:]).mean(0)
			else:
				mean = (self.parameter_history[self.alpha_index(i)].values[burnin:,:]).mean(0)
			std = self.parameter_history[self.alpha_index(i)].values[burnin:,:].std(0)
			plt.plot(self.sample_x,mean,color=colors[i])
			plt.fill_between(self.sample_x[:,0],mean-2*std,mean+2*std,alpha=.2,color=colors[i])
		# [plt.plot(self.sample_x,(self.parameter_history[self.mu_index()].values + self.parameter_history[self.alpha_index(i)].values).mean(0)) for i in range(self.k)]

		if plot_mean:
			mean = self.parameter_history[self.mu_index()].values[burnin:,:].mean(0)
			std = self.parameter_history[self.mu_index()].values[burnin:,:].std(0)
			plt.plot(self.sample_x,mean,'k')
			plt.fill_between(self.sample_x[:,0],mean-2*std,mean+2*std,alpha=.2,color='k')

	def plot_data(self,alpha=1,offset=1):
		import matplotlib.pyplot as plt

		colors = [u'b', u'g', u'r', u'c', u'm', u'y']
		cmaps = [plt.get_cmap(c) for c in ["Blues",'Greens','Reds']]

		# first effect is color,
		# second effect is subplot

		for i in range(self.y.shape[1]):
			if self.k >= 2:
				plt.subplot(1,self.mk[1],self.effect[i,1]+1)
				# c = cmaps[self.effect[i,0]](1.*(self.effect[i,1] + offset)/(self.mk[1]+2*offset))
			# else:
			c = colors[self.effect[i,0]]

			plt.plot(self.x,self.y[:,i],color=c,alpha=alpha)
			plt.ylim(self.y.min(),self.y.max())
