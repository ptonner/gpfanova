import pandas as pd
import numpy as np
import GPy, scipy, time
import matplotlib.pyplot as plt

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

		self.nk = dict([(i,[sum(self.effect[:,i] == j) for j in range(self.mk[i])]) for i in range(self.k)]) # number of observations for each effect level
		if self.k >= 2:
			for j in range(self.mk[0]):
				for l in range(self.mk[1]):
					self.nk[(j,l)] = sum(np.all(self.effect[:,[0,1]]==(j,l),1))

		self.nt = self.y.shape[1]

		# alpha_cols = []
		# for i in range(self.k):
		# 	alpha_cols = alpha_cols + ['alpha_%d(%lf)'%(i,z) for z in self.sample_x]

		# build the index for the parameter cache
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
		ind += ['log_likelihood']

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

		# hack out a static inverse covariance matrix, for speedtests
		# self.k_effect_inv = np.linalg.inv(self.effect_k(i).K(self.sample_x) + np.eye(self.sample_x.shape[0])*1e-9)

	def offset(self):
		"""offset for the calculation of covariance matrices inverse"""
		return 1e-5

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
		if self.k > 1:
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
		"""lth sample of kth effect"""
		return ['%s_%d(%lf)'%(GP_FANOVA.EFFECT_SUFFIXES[k],l,z) for z in self.sample_x]

	def effect_interaction_index(self,k,l,m,n):
		if k < m:
			t1,t2 = k,l
			k,l = m,n
			m,n = t1,t2

		return ['%s_%d:%s_%d(%lf)'%(GP_FANOVA.EFFECT_SUFFIXES[k],l,GP_FANOVA.EFFECT_SUFFIXES[m],n,z) for z in self.sample_x]

	def mu_index(self):
		return ['mu(%lf)'%z for z in self.sample_x]

	def alpha_index(self,k,):
		return ['alpha_%d(%lf)'%(k,z) for z in self.sample_x]

	def likelihood(self,eval=True):
		n = self.n*self.nt
		# mu,cov = np.zeros(n), self.y_k().K(np.tile(self.x,self.nt).T.ravel()[:,None])
		mu = np.zeros(n)

		for i in range(self.nt):
			mu[i*self.n:(i+1)*self.n] += self.parameter_cache[self.mu_index()]

			for k in range(self.k):
				mu[i*self.n:(i+1)*self.n] += self.parameter_cache[self.effect_index(k,self.effect[i,k])]

				# add interaction

		if eval:
			y = self.y.T.ravel()

			prod = 0
			for i in range(y.shape[0]):
				pdf = scipy.stats.norm.logpdf(y[i],loc=mu[i],scale=np.sqrt(self.parameter_cache['y_sigma']))
				prod += pdf

			return prod# norm.pdf(y)

		return mu,cov

	def mu_likelihood(self):
		mu = np.zeros(self.sample_n)
		cov = self.mu_k().K(self.sample_x) + np.eye(self.sample_n)*self.offset()
		return scipy.stats.multivariate_normal.logpdf(self.parameter_cache[self.mu_index()],mu,cov)

	def effect_likelihood(self,i):
		ll = 1
		for j in range(self.mk[i]):
			mu,cov = self.effect_parameters(i,j)

			try:
				ll += scipy.stats.multivariate_normal.logpdf(self.parameter_cache[self.effect_index(i,j)],mu,cov)
			except np.linalg.LinAlgError:
				print i,j

		return ll

	def interaction_likelihood(self,i,k):
		ll = 1
		for j in range(self.mk[i]):
			for l in range(self.mk[k]):
				mu,cov = self.interaction_parameters(i,j,k,l)

				try:
					ll += scipy.stats.multivariate_normal.logpdf(self.parameter_cache[self.effect_interaction_index(i,j,k,l)],mu,cov)
				except np.linalg.LinAlgError:
					print i,j

		return ll

	def y_sub_alpha(self):
		ysa = self.y - np.column_stack([self.parameter_cache[self.alpha_index(i)] for i in self.effect])
		return np.sum(ysa,1)/self.nt

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

		y_k_inv = np.linalg.inv(self.y_k().K(self.sample_x))

		# A = np.linalg.inv(self.mu_k().K(self.sample_x) + offset) + self.nt * np.linalg.inv(self.y_k().K(self.sample_x))
		# b = self.nt*np.dot(np.linalg.inv(self.y_k().K(self.sample_x)),self.y_sub_alpha())

		A = np.linalg.inv(self.mu_k().K(self.sample_x) + offset) + self.nt * y_k_inv
		b = self.nt*np.dot(y_k_inv,self.y_sub_effects())

		A_inv = np.linalg.inv(A)
		return np.dot(A_inv,b), A_inv

	def effect_parameters(self,i,j):
		mu = np.zeros(self.sample_n)
		for k in range(j):
			mu -= self.parameter_cache[self.effect_index(i,k)]
		mu /= (self.mk[i] - j)

		# if j == self.mk[i] - 1: # we're fuckin dun
		# 	return mu,np.zeros((self.sample_n,self.sample_n))

		cov = 1.*(self.mk[i]-j-1)/(self.mk[i]-j) * self.effect_k(i).K(self.sample_x) + np.eye(self.sample_n)*self.offset()

		return mu,cov

	def interaction_parameters(self,i,j,k,l):
		sub_effect_1,sub_effect_2 = np.zeros(self.sample_n), np.zeros(self.sample_n)

		for m in range(j):
			sub_effect_1 -= self.parameter_cache[self.effect_interaction_index(i,m,k,l)]
		for n in range(l):
			sub_effect_2 -= self.parameter_cache[self.effect_interaction_index(i,j,k,n)]

		# print sub_effect_1, sub_effect_2

		# if j == self.mk[i] - 1:
		# 	return sub_effect_1,np.zeros((self.sample_n,self.sample_n))
		# elif l == self.mk[k] - 1:
		# 	return sub_effect_2,np.zeros((self.sample_n,self.sample_n))
		# else:
		mu = - sub_effect_1 - sub_effect_2

		mu = mu/(self.mk[i]-j)/(self.mk[k]-l)
		cov = 1.*(self.mk[i]-j-1)/(self.mk[i]-j)*(self.mk[k]-l-1)/(self.mk[k]-l) * self.effect_interaction_k(i,k).K(self.sample_x) + np.eye(self.sample_n)*self.offset()

		return mu,cov

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
		# k_effect_inv = self.k_effect_inv*1.*(self.mk[i] - j)/(self.mk[i]-j-1)

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

		mu = mu/(self.mk[i]-j)/(self.mk[k]-l)
		cov = 1.*(self.mk[i]-j-1)/(self.mk[i]-j)*(self.mk[k]-l-1)/(self.mk[k]-l) * self.effect_interaction_k(i,k).K(self.sample_x) + np.eye(self.sample_n)*self.offset()

		cov_inv = np.linalg.inv(cov)

		# print mu,cov

		y_k_inv = np.linalg.inv(self.y_k().K(self.sample_x))

		# A = cov + self.nk[i][j] * self.nk[k][l] * y_k_inv
		# b = self.nk[i][j]*self.nk[k][l]*np.dot(y_k_inv,self.y_sub_mu_for_interaction(i,j,k,l)) + np.dot(cov,mu)

		A = cov_inv + self.nk[(j,l)] * y_k_inv
		b = self.nk[(j,l)]*np.dot(y_k_inv,self.y_sub_mu_for_interaction(i,j,k,l)) + np.dot(cov_inv,mu)

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

	def metroplis_hastings_sample(self,_min,_max,delta,parameter,likelihood):
		j1 = scipy.stats.uniform(max(_min,self.parameter_cache[parameter]-delta), min(_max,self.parameter_cache[parameter]+delta))
		new_param = j1.rvs()
		j2 = scipy.stats.uniform(max(_min,new_param-delta), min(_max,new_param+delta))

		old_param = self.parameter_cache[parameter]
		oldll = likelihood()
		self.parameter_cache[parameter] = new_param
		newll = likelihood()

		logr = newll + j2.logpdf(old_param) - oldll - j1.logpdf(new_param)
		r = np.exp(logr)

		if r < 1 and scipy.stats.uniform.rvs(0,1)>max(0,r): # put old back in
			self.parameter_cache[parameter] = old_param

	def mu_metropolis_hastings(self):

		# lengthscale
		_min,_max,delta = 1e-6,100,10
		j1 = scipy.stats.uniform(max(_min,self.parameter_cache['mu_lengthscale']-delta), min(_max,self.parameter_cache['mu_lengthscale']+delta))
		ls = j1.rvs()
		j2 = scipy.stats.uniform(max(_min,ls-delta), min(_max,ls+delta))

		oldls = self.parameter_cache['mu_lengthscale']
		oldll = self.mu_likelihood()
		self.parameter_cache['mu_lengthscale'] = ls
		newll = self.mu_likelihood()

		logr = newll + j2.logpdf(oldls) - oldll - j1.logpdf(ls)
		r = np.exp(logr)

		if r < 1 and scipy.stats.uniform.rvs(0,1)>max(0,r): # put old back in
			self.parameter_cache['mu_lengthscale'] = oldls

	def update(self):
		"""Sample parameters from posterior."""

		# update mu
		mu,cov = self.mu_conditional()
		sample = scipy.stats.multivariate_normal.rvs(mu,cov)
		self.parameter_cache.loc[self.mu_index()] = sample
		# print self.parameter_cache.loc[self.mu_index()]

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
		# likelihood
		self.metroplis_hastings_sample(1e-6,100,1,'y_sigma',self.likelihood)

		# mu
		self.metroplis_hastings_sample(1e-6,100,10,'mu_lengthscale',self.mu_likelihood)
		self.metroplis_hastings_sample(1e-6,100,10,'mu_sigma',self.mu_likelihood)

		# effects
		for i in range(self.k):
			self.metroplis_hastings_sample(1e-6,100,10,"%s_lengthscale"%GP_FANOVA.EFFECT_SUFFIXES[i],lambda: self.effect_likelihood(i))
			self.metroplis_hastings_sample(1e-6,100,10,"%s_sigma"%GP_FANOVA.EFFECT_SUFFIXES[i],lambda: self.effect_likelihood(i))

		# interactions
		for i in range(self.k):
			for k in range(i+1,self.k):
				self.metroplis_hastings_sample(1e-6,100,10,"%s:%s_lengthscale"%(GP_FANOVA.EFFECT_SUFFIXES[i],GP_FANOVA.EFFECT_SUFFIXES[k]),lambda: self.interaction_likelihood(i,k))
				self.metroplis_hastings_sample(1e-6,100,10,"%s:%s_sigma"%(GP_FANOVA.EFFECT_SUFFIXES[i],GP_FANOVA.EFFECT_SUFFIXES[k]),lambda: self.interaction_likelihood(i,k))

		# store the likelihood
		self.parameter_cache['log_likelihood'] = self.likelihood()

	def store(self):
		self.parameter_history = self.parameter_history.append(self.parameter_cache,ignore_index=True)
		self.parameter_history.index = range(self.parameter_history.shape[0])

	def sample(self,n=1,save=0,verbose=False):
		start = self.parameter_history.shape[0]
		i = 1

		start_time = iter_time = time.time()
		while self.parameter_history.shape[0] - start < n:
			self.update()

			if save == 0 or i % save == 0:
				self.store()

				if verbose:
					j = self.parameter_history.shape[0] - start

					print "%d/%d iterations (%.2lf%s) finished in %.2lf minutes" % (j,n,100.*j/n,'%',(time.time()-start_time)/60)
					iter_time = time.time()

			i+=1

		if verbose:
			print "%d samples finished in %.2lf minutes" % (n, (time.time() - start_time)/60)

	def plot_functions(self,plot_mean=True,offset=True,burnin=0):
		import matplotlib.pyplot as plt

		# plt.gca().set_color_cycle(None)
		colors = [u'b', u'g', u'r', u'c', u'm', u'y',]
		cmaps = ["Blues",'Greens','Reds']

		for i in range(self.k):

			for j in range(self.mk[i]):
				if offset:
					mean = (self.parameter_history[self.mu_index()].values[burnin:,:] + self.parameter_history[self.effect_index(i,j)].values[burnin:,:]).mean(0)
				else:
					mean = (self.parameter_history[self.effect_index(i,j)].values[burnin:,:]).mean(0)
				std = self.parameter_history[self.alpha_index(i)].values[burnin:,:].std(0)
				plt.plot(self.sample_x,mean,color=colors[j+sum(self.mk[:i])])
				plt.fill_between(self.sample_x[:,0],mean-2*std,mean+2*std,alpha=.2,color=colors[j+sum(self.mk[:i])])
		# [plt.plot(self.sample_x,(self.parameter_history[self.mu_index()].values + self.parameter_history[self.alpha_index(i)].values).mean(0)) for i in range(self.k)]

		if plot_mean:
			mean = self.parameter_history[self.mu_index()].values[burnin:,:].mean(0)
			std = self.parameter_history[self.mu_index()].values[burnin:,:].std(0)
			plt.plot(self.sample_x,mean,'k')
			plt.fill_between(self.sample_x[:,0],mean-2*std,mean+2*std,alpha=.2,color='k')

	def plot_data(self,alpha=1,offset=1):

		colors = [u'b', u'g', u'r', u'c', u'm', u'y']
		cmaps = [plt.get_cmap(c) for c in ["Blues",'Greens','Reds']]

		# first effect is color,
		# second effect is subplot

		if self.mk[0] <= len(colors):
			for i in range(self.y.shape[1]):
				if self.k >= 2:
					plt.subplot(1,self.mk[1],self.effect[i,1]+1)
					# c = cmaps[self.effect[i,0]](1.*(self.effect[i,1] + offset)/(self.mk[1]+2*offset))
				# else:
				c = colors[self.effect[i,0]]

				plt.plot(self.x,self.y[:,i],color=c,alpha=alpha)
				plt.ylim(self.y.min(),self.y.max())
		else:
			for i in range(self.mk[0]):
				y = self.y[:,self.effect[:,0]==i]
				plt.plot(self.x,y.mean(1),c='k',alpha=alpha)
				plt.fill_between(self.x[:,0],y.mean(1)-y.std(1),y.mean(1)+y.std(1),color='k',alpha=.2)


	def plot_two_effects(self,i,k,_data=True,function=False,samples=True,offset=True,subplot_dims=None,data_plot_kwargs={}):
		colors = [u'b', u'g', u'r', u'c', u'm', u'y']

		if subplot_dims is None:
			subplot_dims = range(2)

		if i in subplot_dims:
			nrows = self.mk[i]
		else:
			nrows = 1
		if k in subplot_dims:
			ncols = self.mk[k]
		else:
			ncols = 1


		for j in range(self.mk[i]):
			for l in range(self.mk[k]):

				if i in subplot_dims and k in subplot_dims:
					pos = self.mk[k]*j + l+1
					c = 'k'
				elif i in subplot_dims:
					pos = j+1
					c = colors[l]
				elif k in subplot_dims:
					pos = l+1
					c = colors[j]
				else:
					pos = 1
					c = colors[j*self.mk[i]+l]
				plt.subplot(nrows,ncols,pos)

				if i in subplot_dims and k in subplot_dims:
					plt.title("%d, %d" % (j,l))

				if _data:
					data = self.y[:,np.all(self.effect[:,[i,k]]==(j,l),1)]
					mean = data.mean(1)
					std = data.std(1)

					if samples:
						plt.plot(self.x[:,0],data,c,**data_plot_kwargs)
					else:
						plt.plot(self.x[:,0],mean,c)
						plt.fill_between(self.x[:,0],mean-2*std,mean+2*std,color=c,alpha=.2)

				if function:
					data = self.parameter_history[self.effect_index(i,j)].values + self.parameter_history[self.effect_index(k,l)].values + self.parameter_history[self.effect_interaction_index(k,l,i,j)].values
					if offset:
						data += self.parameter_history[self.mu_index()].values
					mean = data.mean(0)
					std = data.std(0)

					plt.plot(self.sample_x,mean,c)
					plt.fill_between(self.sample_x[:,0],mean-2*std,mean+2*std,color=c,alpha=.2)

				plt.ylim(self.y.min(),self.y.max())

				if function and not offset:
					plt.ylim(mean.min()-std.max()*2,mean.max()+std.max()*2)

	def plot_effect(self,subplots=True,title_kwargs={}):
		for i in range(self.k):
			for j in range(self.mk[i]):

				if subplots:
					plt.subplot(max(self.mk),self.k,i+1+self.k*j)
					plt.title("$\\%s_%d$"%(GP_FANOVA.EFFECT_SUFFIXES[i],j),**title_kwargs)

				data = self.parameter_history[self.effect_index(i,j)].values
				# if offset:
				# 	data += self.parameter_history[self.mu_index()].values
				mean = data.mean(0)
				std = data.std(0)

				plt.plot(self.sample_x,mean)
				plt.fill_between(self.sample_x[:,0],mean-2*std,mean+2*std,alpha=.2)

	def plot_hyper_params(self):

		def plot_param(param):
			plt.plot(self.parameter_history[param])
			plt.title(param,fontsize=20)

		plt.subplot(5,2,1)
		plot_param('mu_lengthscale')
		plt.subplot(5,2,2)
		plot_param('mu_sigma')

		plt.subplot(5,2,3)
		plot_param('alpha_lengthscale')
		plt.subplot(5,2,4)
		plot_param('alpha_sigma')

		plt.subplot(5,2,5)
		plot_param('beta_lengthscale')
		plt.subplot(5,2,6)
		plot_param('beta_sigma')

		plt.subplot(5,2,7)
		plot_param('alpha:beta_lengthscale')
		plt.subplot(5,2,8)
		plot_param('alpha:beta_sigma')

		plt.subplot(5,2,9)
		plot_param('y_sigma')
