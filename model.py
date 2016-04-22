import pandas as pd
import numpy as np
import GPy, scipy

class GP_FANOVA(object):

	EFFECT_SUFFIXES = ['alpha','beta','gamma']

	def __init__(self,x,y,effect):
		"""

		x: (n,p)
		y: (n,n1+n2+...)
		effect: (n1+n2+...)
		"""

		self.x = x
		self.y = y
		self.effect = effect
		self.sample_x = np.unique(self.x)[:,None] # this only works for one dimension

		self.n = x.shape[0]
		# assert self.n == self.y.shape[0], 'x and y must be same length'

		self.sample_n = self.sample_x.shape[0]

		self.p = x.shape[1]

		self.k = self.effect.shape[1] # number of effects
		self.mk = [np.unique(self.effect[:,i]).shape[0] for i in range(self.k)] # number of levels for each effect
		# self.nk = {[(i,[sum(self.effect[:,i] == j) for j in range(self.mk[i])]) for i in range(self.k)]} # number of observations for each effect level
		self.nt = self.y.shape[1]

		alpha_cols = []
		for i in range(self.k):
			alpha_cols = alpha_cols + ['alpha_%d(%lf)'%(i,z) for z in self.sample_x]

		ind = self.mu_index() + ['mu_sigma','mu_lengthscale']
		for i in range(self.k):
			for j in range(self.mk[i]):
				ind += self.effect_index(i,j)
			ind += ["%s_sigma"%GP_FANOVA.EFFECT_SUFFIXES[i],"%s_lengthscale"%GP_FANOVA.EFFECT_SUFFIXES[i]]

			# add interaction samples
			for k in range(i):
				for l in range(self.mk[i]):
					for m in range(self.mk[k]):
						ind += self.effect_interaction_index(i,l,k,m)
				ind += ["%s:%s_sigma"%(GP_FANOVA.EFFECT_SUFFIXES[k],GP_FANOVA.EFFECT_SUFFIXES[i]),
						"%s:%s_lengthscale"%(GP_FANOVA.EFFECT_SUFFIXES[k],GP_FANOVA.EFFECT_SUFFIXES[i])]

		ind += ['y_sigma','y_lengthscale']

		# print self.sample_n*(1+sum(self.mk))+4+self.k*2, len(ind)
		self.parameter_cache = pd.Series(np.zeros(len(ind)),
											index=ind)

		self.parameter_cache[['mu_sigma','mu_lengthscale','y_sigma','y_lengthscale']] = 1
		for i in range(self.k):
			self.parameter_cache[['%s_sigma'%GP_FANOVA.EFFECT_SUFFIXES[i],'%s_lengthscale'%GP_FANOVA.EFFECT_SUFFIXES[i]]] = 1
			for j in range(i):
				self.parameter_cache[['%s:%s_sigma'%(GP_FANOVA.EFFECT_SUFFIXES[j],GP_FANOVA.EFFECT_SUFFIXES[i]),'%s:%s_lengthscale'%(GP_FANOVA.EFFECT_SUFFIXES[j],GP_FANOVA.EFFECT_SUFFIXES[i])]] = 1
		self.parameter_cache['y_sigma'] = .1

		self.parameter_history = pd.DataFrame(columns=ind)

		# mu and alpha samples, kernel hyperparams (mu, alpha, noise)
		# self.parameter_history = pd.DataFrame(np.zeros(self.sample_n*(1+self.k)+6)[None,:],
		# 							columns=['mu(%lf)'%z for z in self.sample_x]+
		# 									alpha_cols+
		# 									['mu_sigma','mu_lengthscale',
		# 									'alpha_sigma','alpha_lengthscale',
		# 									'y_sigma','y_lengthscale'])
		# self.parameter_history.iloc[0,-6:] = 1
		# self.parameter_history.loc[0,['y_sigma']] = .1
		#
		# self.parameter_cache = self.parameter_history.iloc[0,:]

	def offset(self):
		return 1e-9

	def sample_prior(self,update_data=False):
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

	def effect_index(self,k,l,):
		return ['%s_%d(%lf)'%(GP_FANOVA.EFFECT_SUFFIXES[k],l,z) for z in self.sample_x]

	def effect_interaction_index(self,k,l,m,n):
		return ['%s_%d:%s_%d(%lf)'%(GP_FANOVA.EFFECT_SUFFIXES[k],l,GP_FANOVA.EFFECT_SUFFIXES[m],n,z) for z in self.sample_x]

	def mu_index(self):
		return ['mu(%lf)'%z for z in self.sample_x]

	def alpha_index(self,k,):
		return ['alpha_%d(%lf)'%(k,z) for z in self.sample_x]

	def y_sub_alpha(self):
		ysa = self.y - np.column_stack([self.parameter_cache[self.alpha_index(i)] for i in self.effect])
		return np.sum(ysa,1)/self.nt

	def y_sub_mu(self,i):
		ysm = self.y[:,self.effect==i] - self.parameter_cache[self.mu_index()].values[:,None]
		return np.sum(ysm,1)/self.nk[i]

	def mu_k(self):
		sigma,ls = self.parameter_cache[['mu_sigma','mu_lengthscale']]
		return GPy.kern.RBF(self.p,variance=sigma,lengthscale=ls)

	def effect_k(self,i):
		sigma,ls = self.parameter_cache[["%s_sigma"%GP_FANOVA.EFFECT_SUFFIXES[i],"%s_lengthscale"%GP_FANOVA.EFFECT_SUFFIXES[i]]]
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
		b = self.nt*np.dot(np.linalg.inv(self.y_k().K(self.sample_x)),self.y_sub_alpha())

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

		# update alpha
		# order = np.

		order = np.random.choice(range(self.k),self.k,replace=False)
		for i in order:
			mu,cov = self.alpha_conditional(i,order)
			sample = scipy.stats.multivariate_normal.rvs(mu,cov)
			self.parameter_cache.loc[self.alpha_index(i)] = sample

		# update hyperparams

	def store(self):
		self.parameter_history = self.parameter_history.append(self.parameter_cache)
		self.parameter_history.index = range(self.parameter_history.shape[0])

	def sample(self,n=1,save=0):
		start = self.parameter_history.index[-1]
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

		for i in range(self.y.shape[1]):
			if self.k == 2:
				c = cmaps[self.effect[i,0]](1.*(self.effect[i,1] + offset)/(self.mk[1]+2*offset))
			else:
				c = colors(self.effect[i,0])

			plt.plot(self.x,self.y[:,i],color=c,alpha=alpha)
