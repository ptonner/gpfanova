from patsy.contrasts import Sum
from sample import SamplerContainer, Gibbs, Slice, Fixed, Transform
from kernel import RBF, White
import numpy as np
import GPy, scipy

class GP_FANOVA(SamplerContainer):

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

		# SamplerContainer
		# samplers = [Fixed('y_sigma','y_sigma',)]
		samplers = [Slice('y_sigma','y_sigma',self.y_likelihood,.1,10)]
		samplers += [Gibbs('mu',self.mu_index(),self.mu_conditional_params)]
		samplers += [Slice('mu_sigma','mu_sigma',self.mu_likelihood,.1,10)]
		samplers += [Slice('mu_lengthscale','mu_lengthscale',self.mu_likelihood,.1,10)]
		for i in range(self.k):
			for j in range(self.mk[i]-1):
				samplers.append(Gibbs('%s*_%d'%(GP_FANOVA.EFFECT_SUFFIXES[i],j),
										self.effect_contrast_index(i,j),
										lambda i=i,j=j : self.effect_contrast_conditional_params(i,j)))
			samplers += [Slice('%s*_sigma'%GP_FANOVA.EFFECT_SUFFIXES[i],'%s*_sigma'%GP_FANOVA.EFFECT_SUFFIXES[i],lambda x: self.effect_contrast_likelihood(i=i,sigma=x),.1,10)]
			samplers += [Slice('%s*_lengthscale'%GP_FANOVA.EFFECT_SUFFIXES[i],'%s*_lengthscale'%GP_FANOVA.EFFECT_SUFFIXES[i],lambda x: self.effect_contrast_likelihood(i=i,lengthscale=x),.1,10)]

		# add effect transforms
		for k in range(self.k):
			for l in range(self.mk[i]):
				samplers.append(Transform('%s_%d'%(GP_FANOVA.EFFECT_SUFFIXES[k],l),
										self.effect_index(k,l),
										lambda k=k,l=l : self.effect_sample(k,l)))

		SamplerContainer.__init__(self,*samplers)

		# contrasts
		self.contrasts = [self.effect_contrast_matrix(i) for i in range(self.k)]

		# kenels
		self.y_k = White(self,['y_sigma'],logspace=True)
		self.mu_k = RBF(self,['mu_sigma','mu_lengthscale'],logspace=True)
		self._effect_contrast_k = [RBF(self,['%s*_sigma'%GP_FANOVA.EFFECT_SUFFIXES[i],'%s*_lengthscale'%GP_FANOVA.EFFECT_SUFFIXES[i]],logspace=True) for i in range(self.k)]

	def offset(self):
		"""offset for the calculation of covariance matrices inverse"""
		return 1e-9

	def effect_index(self,k,l,):
		"""lth sample of kth effect"""
		return ['%s_%d(%lf)'%(GP_FANOVA.EFFECT_SUFFIXES[k],l,z) for z in self.x]

	def effect_contrast_index(self,k,l,):
		"""lth sample of kth effect"""
		return ['%s*_%d(%lf)'%(GP_FANOVA.EFFECT_SUFFIXES[k],l,z) for z in self.x]

	def mu_index(self):
		return ['mu(%lf)'%z for z in self.x]

	def effect_contrast_k(self,i):

		return self._effect_contrast_k[i]

	def effect_contrast_matrix(self,i):
		h = Sum().code_without_intercept(range(self.mk[i])).matrix

		return h

	def y_k_inv(self,x=None):
		if x is None:
			x = self.x

		return self.y_k.K_inv(x)

	def mu_k_inv(self,x=None):
		if x is None:
			x = self.x

		return self.mu_k.K_inv(x)

	def contrast_k_inv(self,i,x=None):
		if x is None:
			x = self.x

		return self._effect_contrast_k[i].K_inv(x)

	def effect_contrast_array(self,i,history=None,deriv=False):

		if deriv:
			loc = self.derivative_history
		elif not history is None:
			loc = self.parameter_history
		else:
			loc = self.parameter_cache

		a = np.zeros((self.n,self.mk[i]-1))
		for j in range(self.mk[i]-1):
			if history is None:
				a[:,j] = loc[self.effect_contrast_index(i,j)]
			else:
				a[:,j] = loc.loc[history,self.effect_contrast_index(i,j)]
		return a

	def mu_conditional_params(self,history=None,cholesky=True,m_inv=None,y_inv=None):
		m = np.zeros(self.n)
		obs = np.zeros(self.n) # number of observations at each timepoint

		y_effect = np.zeros((self.n,self.r))
		for r in range(self.r):
			obs += 1 # all timepoints observed, need to update for nan's
			for i in range(self.k):
				y_effect[:,r] = self.y[:,r] - np.dot(self.effect_contrast_array(i,history), self.contrasts[i][self.effect[r,i]])

				# need to do interaction here

		m = np.mean(y_effect,1)

		if cholesky:

			if m_inv is None:
				m_inv = self.mu_k_inv()
			if y_inv is None:
				y_inv = self.y_k_inv()

			A = obs*y_inv + m_inv
			b = obs*np.dot(y_inv,m)

			chol_A = np.linalg.cholesky(A)
			chol_A_inv = np.linalg.inv(chol_A)
			A_inv = np.dot(chol_A_inv.T,chol_A_inv)

		else:
			obs_cov_inv = np.linalg.inv(self.y_k.K(self.x))

			A = obs*obs_cov_inv + np.linalg.inv(self.mu_k().K(self.x) + np.eye(self.n)*self.offset())
			b = obs*np.dot(obs_cov_inv,m)

			A_inv = np.linalg.inv(A)
		return np.dot(A_inv,b), A_inv

	def effect_contrast_conditional_params(self,i,j,cholesky=True,c_inv=None,y_inv=None):
		"""compute the conditional mean and covariance of an effect contrast function"""
		m = np.zeros(self.n)
		obs = np.zeros(self.n) # number of observations at each timepoint

		contrasts = self.effect_contrast_array(i)

		tot = 0
		# calculate residual for each replicate, r
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

		if cholesky:
			if c_inv is None:
				c_inv = self.contrast_k_inv(i)

			if y_inv is None:
				y_inv = self.y_k_inv()

			A = obs*y_inv + c_inv
			b = obs*np.dot(y_inv,m)

			chol_A = np.linalg.cholesky(A)
			chol_A_inv = np.linalg.inv(chol_A)
			A_inv = np.dot(chol_A_inv.T,chol_A_inv)

		else:
			obs_cov_inv = np.linalg.inv(self.y_k.K(self.x))

			A = obs_cov_inv*obs + np.linalg.inv(self.effect_contrast_k(i).K(self.x) + np.eye(self.n)*self.offset())
			b = obs*np.dot(obs_cov_inv,m)

			A_inv = np.linalg.inv(A)
		return np.dot(A_inv,b), A_inv

	def effect_sample(self,i,j):

		return np.dot(self.effect_contrast_array(i),self.contrasts[i][j,:])

	def y_mu(self):
		mu = np.zeros(self.n*self.r)
		for i in range(self.r):
			mu[i*self.n:(i+1)*self.n] += self.parameter_cache[self.mu_index()]
			for k in range(self.k):
				for l in range(self.mk[k]-1):
					mu[i*self.n:(i+1)*self.n] += self.effect_sample(k,l)

		return mu

	def y_likelihood(self,sigma=None):
		y = np.ravel(self.y.T)
		mu = self.y_mu()
		sigma = 10**sigma

		return np.sum(scipy.stats.norm.logpdf(y-mu,0,sigma))

	def mu_likelihood(self,sigma=None,ls=None):
		mu = np.zeros(self.n)
		# cov = self.mu_k(sigma=sigma,ls=ls).K(self.x)
		cov = self.mu_k.K(self.x,sigma,ls)
		cov += cov.mean()*np.eye(self.n)*1e-6
		return scipy.stats.multivariate_normal.logpdf(self.parameter_cache[self.mu_index()],mu,cov)

	def effect_contrast_likelihood(self,i,sigma=None,lengthscale=None):
		ll = 1
		for j in range(self.mk[i]-1):
			mu = np.zeros(self.n)
			cov = self.effect_contrast_k(i).K(self.x,sigma,lengthscale) + np.eye(self.n)*1e-6

			try:
				ll += scipy.stats.multivariate_normal.logpdf(self.parameter_cache[self.effect_contrast_index(i,j)],mu,cov)
			except np.linalg.LinAlgError:
				print "likelihood LinAlgError (%d,%d)" % (i,j)

		return ll
