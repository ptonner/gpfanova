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

	def mu_index(self):
		return ['mu(%lf)'%z for z in self.sample_x]

	def alpha_index(self,k,):
		return ['alpha_%d(%lf)'%(k,z) for z in self.sample_x]

	def y_sub_alpha(self):
		ysa = self.y - np.column_stack([self.parameter_history.loc[self.parameter_history.index[-1],self.alpha_index(i)].T for i in self.effect])
		return np.sum(ysa,1)/self.nt

	def y_sub_mu(self,i):
		ysm = self.y[:,self.effect==i] - self.parameter_history.loc[self.parameter_history.index[-1],self.mu_index()].T.values[:,None]
		return np.sum(ysm,1)/self.nk[i]

	def mu_k(self):
		sigma,ls = self.parameter_history[['mu_sigma','mu_lengthscale']].iloc[-1,:]
		return GPy.kern.RBF(self.p,variance=sigma,lengthscale=ls)

	def alpha_k(self):
		sigma,ls = self.parameter_history[['alpha_sigma','alpha_lengthscale']].iloc[-1,:]
		return GPy.kern.RBF(self.p,variance=sigma,lengthscale=ls)

	def y_k(self):
		sigma,ls = self.parameter_history[['y_sigma','y_lengthscale']].iloc[-1,:]
		# return GPy.kern.RBF(self.p,variance=sigma,lengthscale=ls)
		return GPy.kern.White(self.p,variance=sigma)

	def mu_conditional(self):
		A = self.mu_k().K(self.sample_x) + self.nt * np.linalg.inv(self.y_k().K(self.sample_x))
		b = self.nt*np.dot(np.linalg.inv(self.y_k().K(self.sample_x)),self.y_sub_alpha())

		A_inv = np.linalg.inv(A)
		return np.dot(A_inv,b), A_inv

	def alpha_conditional(self,i):
		if i == self.k-1: # enforce sum to zero constraint
			mu_alpha = np.zeros(self.sample_n)
			for j in range(i):
				mu_alpha = mu_alpha - self.parameter_history[self.alpha_index(j)].iloc[-1,:].values
			return mu_alpha,np.zeros((self.sample_n,self.sample_n))

		y_k_inv = np.linalg.inv(self.y_k().K(self.sample_x))
		k_alpha_inv = np.linalg.inv(1.*(self.k-i-1)/(self.k-i) * self.alpha_k().K(self.sample_x))

		A = k_alpha_inv + self.nk[i] * y_k_inv

		mu_alpha = np.zeros(self.sample_n)
		for j in range(i):
			mu_alpha = mu_alpha - self.parameter_history[self.alpha_index(j)].iloc[-1,:].values
		mu_alpha = mu_alpha / (self.k - i)

		b = self.nk[i]*np.dot(y_k_inv,self.y_sub_mu(i)) + np.dot(k_alpha_inv,mu_alpha)

		A_inv = np.linalg.inv(A)
		return np.dot(A_inv,b), A_inv

	def update(self):

		new_params = pd.DataFrame(self.parameter_history.iloc[-1,:]).T
		new_params.index = [self.parameter_history.shape[0]]

		self.parameter_history = self.parameter_history.append(new_params)

		# update mu

		# update alpha
		for i in range(self.k):
			print i
			mu,cov = self.alpha_conditional(i)
			sample = scipy.stats.multivariate_normal.rvs(mu,cov)
			self.parameter_history.loc[self.parameter_history.index[-1],self.alpha_index(i)] = sample

		# update hyperparams

	def sample(n=1):
		for i in range(n):
			self.update()
