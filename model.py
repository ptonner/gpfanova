import pandas as pd
import numpy as np
import GPy, scipy

class GP_FANOVA(object):

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
		assert self.n == self.y.shape[0], 'x and y must be same length'

		self.sample_n = self.sample_x.shape[0]

		self.p = x.shape[1]

		self.k = np.unique(self.effect).shape[0]
		self.nk = [sum(self.effect == i) for i in range(self.k)]
		self.nt = self.y.shape[1]

		alpha_cols = []
		for i in range(self.k):
			alpha_cols = alpha_cols + ['alpha_%d(%lf)'%(i,z) for z in self.sample_x]

		# mu and alpha samples, kernel hyperparams (mu, alpha, noise)
		self.parameter_history = pd.DataFrame(np.zeros(self.sample_n*(1+self.k)+6)[None,:],
									columns=['mu(%lf)'%z for z in self.sample_x]+
											alpha_cols+
											['mu_sigma','mu_lengthscale',
											'alpha_sigma','alpha_lengthscale',
											'y_sigma','y_lengthscale'])
		self.parameter_history.iloc[0,-6:] = 1
		self.parameter_history.loc[0,['y_sigma']] = .1

		self.parameter_cache = self.parameter_history.iloc[0,:]

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

	def plot_functions(self,burnin=0):
		import matplotlib.pyplot as plt

		# plt.gca().set_color_cycle(None)
		colors = [u'b', u'g', u'r', u'c', u'm', u'y',]

		for i in range(self.k):
			mean = (self.parameter_history[self.mu_index()].values[burnin:,:] + self.parameter_history[self.alpha_index(i)].values[burnin:,:]).mean(0)
			std = self.parameter_history[self.alpha_index(i)].values[burnin:,:].std(0)
			plt.plot(self.sample_x,mean,color=colors[i])
			plt.fill_between(self.sample_x[:,0],mean-2*std,mean+2*std,alpha=.2,color=colors[i])
		# [plt.plot(self.sample_x,(self.parameter_history[self.mu_index()].values + self.parameter_history[self.alpha_index(i)].values).mean(0)) for i in range(self.k)]

		mean = self.parameter_history[self.mu_index()].values[burnin:,:].mean(0)
		std = self.parameter_history[self.mu_index()].values[burnin:,:].std(0)
		plt.plot(self.sample_x,mean,'k')
		plt.fill_between(self.sample_x[:,0],mean-2*std,mean+2*std,alpha=.2,color='k')

	def plot_data(self):
		import matplotlib.pyplot as plt

		colors = [u'b', u'g', u'r', u'c', u'm', u'y']

		for i in range(self.y.shape[1]):
			plt.plot(self.x,self.y[:,i],color=colors[self.effect[i]],alpha=.6)
