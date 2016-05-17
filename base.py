import pandas as pd
import numpy as np
import GPy, scipy, time
from patsy.contrasts import Helmert, Sum

class GP_FANOVA(object):

	EFFECT_SUFFIXES = ['alpha','beta','gamma','delta','epsilon']

	def __init__(self,x,y,effect,contrast=None):
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

		ind = []
		for i in range(self.k):
			for j in range(self.mk[i]):
				ind += self.effect_index(i,j)

				# interactions
				for k in range(i):
					for l in range(self.mk[i]):
						ind += self.effect_interaction_index(i,j,k,l)

		self.effect_history = pd.DataFrame(columns=ind)

		# contrasts
		if contrast is None:
			self.contrasts = [self.effect_contrast_matrix_sum(i) for i in range(self.k)]
		elif contrast == "helmert":
			self.contrasts = [self.effect_contrast_matrix_helmert(i) for i in range(self.k)]
		else:
			self.contrasts = [self.effect_contrast_matrix_sum(i) for i in range(self.k)]

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
					for m in range(self.mk[k]-1):
						ind += self.effect_interaction_contrast_index(i,l,k,m)
				ind += ["(%s:%s)*_sigma"%(GP_FANOVA.EFFECT_SUFFIXES[k],GP_FANOVA.EFFECT_SUFFIXES[i]),
						"(%s:%s)*_lengthscale"%(GP_FANOVA.EFFECT_SUFFIXES[k],GP_FANOVA.EFFECT_SUFFIXES[i])]

		ind += ['y_sigma','y_lengthscale']

		return ind

	def effect_contrast_matrix_sum(self,i):
		h = Sum().code_without_intercept(range(self.mk[i])).matrix

		return h

	def effect_contrast_matrix_helmert(self,i):
		h = Helmert().code_without_intercept(range(self.mk[i])).matrix
		h /= h.shape[1]

		v = 1.*(h.shape[0]-1)/h.shape[0]

		b = np.zeros(h.shape[1])
		b[-1] = np.sqrt(v)

		ind = range(h.shape[1]-1)
		ind.reverse()

		# compute column scale base on previous column scalings
		for j in ind:
			b[j] = np.sqrt((v-np.dot(h[j+1,:]**2,b**2))/(h[j+1,j]**2))

		h = h*b

		# reverse rows and columns
		ind = range(h.shape[1])
		ind.reverse()
		h = h[:,ind]

		ind = range(h.shape[0])
		ind.reverse()
		h = h[ind,:]

		if not np.allclose(np.sum(h**2,1),[v]*h.shape[0]):
			print 'single row constraint fail! '+str(h)
			return h
		assert np.allclose(np.sum((h[:-1,:]*h[1:,:]),1),[-1./h.shape[0]]*(h.shape[1])), 'cross row constraint fail! '+str(h)

		return h


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

	def effect_index(self,k,l,):
		"""lth sample of kth effect"""
		return ['%s_%d(%lf)'%(GP_FANOVA.EFFECT_SUFFIXES[k],l,z) for z in self.x]

	def effect_interaction_index(self,k,l,m,n):
		if k < m:
			t1,t2 = k,l
			k,l = m,n
			m,n = t1,t2

		return ['%s_%d:%s_%d(%lf)'%(GP_FANOVA.EFFECT_SUFFIXES[k],l,GP_FANOVA.EFFECT_SUFFIXES[m],n,z) for z in self.x]

	def effect_contrast_index(self,k,l,):
		"""lth sample of kth effect"""
		return ['%s*_%d(%lf)'%(GP_FANOVA.EFFECT_SUFFIXES[k],l,z) for z in self.x]

	def effect_interaction_contrast_index(self,k,l,m,n):
		if k < m:
			return effect_interaction_contrast_index(m,n,k,l)

		return ['(%s_%d:%s_%d)*(%lf)'%(GP_FANOVA.EFFECT_SUFFIXES[k],l,GP_FANOVA.EFFECT_SUFFIXES[m],n,z) for z in self.x]

	def mu_index(self):
		return ['mu(%lf)'%z for z in self.x]

	def effect_contrast_array(self,i,history=None):
		a = np.zeros((self.n,self.mk[i]-1))
		for j in range(self.mk[i]-1):
			if history is None:
				a[:,j] = self.parameter_cache[self.effect_contrast_index(i,j)]
			else:
				a[:,j] = self.parameter_history.loc[history,self.effect_contrast_index(i,j)]
		return a

	def effect_contrast_conditional_params(self,i,j):
		"""compute the conditional mean and covariance of an effect contrast function"""
		m = np.zeros(self.n)
		obs = np.zeros(self.n) # number of observations at each timepoint

		contrasts = self.effect_contrast_array(i)

		tot = 0
		for r in range(self.r):
			e = self.effect[r,i]
			if self.contrasts[i][e,j] == 0: # don't use this observation
				continue
			obs += ~np.isnan(self.y[:,r])
			tot += 1

			resid = self.y[:,r] - self.parameter_cache[self.mu_index()] - np.dot(contrasts,self.contrasts[i][e,:])

			# add back in this contrast
			resid += contrasts[:,j] * self.contrasts[i][e,j]

			# scale by contrast value
			resid /= self.contrasts[i][e,j]

			m+= resid
		m /= tot

		obs_cov_inv = np.linalg.inv(self.y_k().K(self.x))

		A = obs_cov_inv*obs + np.linalg.inv(self.effect_contrast_k(i).K(self.x) + np.eye(self.n)*self.offset())
		b = obs*np.dot(obs_cov_inv,m)

		A_inv = np.linalg.inv(A)
		return np.dot(A_inv,b), A_inv

	def mu_conditional_params(self):
		m = np.zeros(self.n)
		obs = np.zeros(self.n) # number of observations at each timepoint

		y_effect = np.zeros((self.n,self.r))
		for r in range(self.r):
			obs += 1 # all timepoints observed, need to update for nan's
			for i in range(self.k):
				y_effect[:,r] = self.y[:,r] - np.dot(self.effect_contrast_array(i), self.contrasts[i][self.effect[r,i]])

				# need to do interaction here

		m = np.mean(y_effect,1)
		obs_cov_inv = np.linalg.inv(self.y_k().K(self.x))

		A = obs*obs_cov_inv + np.linalg.inv(self.mu_k().K(self.x) + np.eye(self.n)*self.offset())
		b = obs*np.dot(obs_cov_inv,m)

		A_inv = np.linalg.inv(A)
		return np.dot(A_inv,b), A_inv

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

	def gibbs_sample(self,param_fxn,parameters):
		mu,cov = param_fxn()
		sample = scipy.stats.multivariate_normal.rvs(mu,cov)
		self.parameter_cache[parameters] = sample

	def update(self):

		# update mu
		self.gibbs_sample(self.mu_conditional_params,self.mu_index())

		for i in range(self.k):
			for j in range(self.mk[i]-1):
				self.gibbs_sample(lambda: self.effect_contrast_conditional_params(i,j),
									self.effect_contrast_index(i,j))

		return

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

	def effect_samples(self,i,j):
		samples = []

		# compute samples not done already
		for r in range(self.effect_history.shape[0],self.parameter_history.shape[0]):
			self.effect_history.loc[r,self.effect_index(i,j)] = np.dot(self.effect_contrast_array(i,r),self.contrasts[i][j,:])

		return self.effect_history[self.effect_index(i,j)].values

	def plot_functions(self,plot_mean=True,offset=True,burnin=0,variance=False):
		import matplotlib.pyplot as plt

		# plt.gca().set_color_cycle(None)
		colors = [u'b', u'g', u'r', u'c', u'm', u'y',]
		cmaps = ["Blues",'Greens','Reds']

		for i in range(self.k):
			for j in range(self.mk[i]):
				samples = self.effect_samples(i,j)[burnin:,:]
				if offset:
					samples += self.parameter_history[self.mu_index()].values[burnin:,:]

				mean = samples.mean(0)
				std = samples.std(0)

				if variance:
					plt.plot(self.x,std,color=colors[j])
				else:
					plt.plot(self.x,mean,color=colors[j])
					plt.fill_between(self.x[:,0],mean-2*std,mean+2*std,alpha=.2,color=colors[j])
		# [plt.plot(self.sample_x,(self.parameter_history[self.mu_index()].values + self.parameter_history[self.alpha_index(i)].values).mean(0)) for i in range(self.k)]

		if plot_mean:
			mean = self.parameter_history[self.mu_index()].values[burnin:,:].mean(0)
			std = self.parameter_history[self.mu_index()].values[burnin:,:].std(0)

			if variance:
				plt.plot(self.x,std,color='k')
			else:
				plt.plot(self.x,mean,'k')
				plt.fill_between(self.x[:,0],mean-2*std,mean+2*std,alpha=.2,color='k')

	def plot_contrasts(self,burnin=0):
		import matplotlib.pyplot as plt

		# plt.gca().set_color_cycle(None)
		colors = [u'b', u'g', u'r', u'c', u'm', u'y',]
		cmaps = ["Blues",'Greens','Reds']

		for i in range(self.k):
			for j in range(self.mk[i]-1):
				samples = self.parameter_history[self.effect_contrast_index(i,j)].values[burnin:,:]

				mean = samples.mean(0)
				std = samples.std(0)

				plt.plot(self.x,mean,color=colors[j])
				plt.fill_between(self.x[:,0],mean-2*std,mean+2*std,alpha=.2,color=colors[j])

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
