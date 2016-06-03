# cython: profile=True

from patsy.contrasts import Sum
from sample import SamplerContainer, Gibbs, Slice, Fixed, Transform
from kernel import RBF, White
import numpy as np
import GPy, scipy, logging

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
		self.m = self.y.shape[1]

		self.r = self.nt = self.y.shape[1]
		assert self.r == self.effect.shape[0], 'y second dimension must match effect first dimension'

		self.k = self.effect.shape[1] # number of effects
		self.mk = [np.unique(self.effect[:,i]).shape[0] for i in range(self.k)] # number of levels for each effect

		# indexes
		self._effect_contrast_index = []
		for k in range(self.k):
			self._effect_contrast_index.append([])
			for l in range(self.mk[k]):
				self._effect_contrast_index[-1].append(['%s*_%d(%lf)'%(GP_FANOVA.EFFECT_SUFFIXES[k],l,z) for z in self.x])

		# kenels
		self.y_k = White(self,['y_sigma'],logspace=True)
		self.mu_k = RBF(self,['mu_sigma','mu_lengthscale'],logspace=True)
		self._effect_contrast_k = [RBF(self,['%s*_sigma'%GP_FANOVA.EFFECT_SUFFIXES[i],'%s*_lengthscale'%GP_FANOVA.EFFECT_SUFFIXES[i]],logspace=True) for i in range(self.k)]

		# SamplerContainer
		samplers = [Slice('y_sigma','y_sigma',self.y_likelihood,.1,10)]
		samplers += [Gibbs('mu',self.mu_index(),lambda f=0,k=self.mu_k: self.function_conditional(f,k))]
		samplers += [Slice('mu_sigma','mu_sigma',self.mu_likelihood,.1,10)]
		samplers += [Slice('mu_lengthscale','mu_lengthscale',self.mu_likelihood,.1,10)]
		for i in range(self.k):
			for j in range(self.mk[i]-1):
				samplers.append(Gibbs('%s*_%d'%(GP_FANOVA.EFFECT_SUFFIXES[i],j),
										self.effect_contrast_index(i,j),
										lambda i=i,j=j,k=self._effect_contrast_k[i] : self.function_conditional((i,j),k)))
				# samplers.append(Gibbs('%s*_%d'%(GP_FANOVA.EFFECT_SUFFIXES[i],j),
				# 						self.effect_contrast_index(i,j),
				# 						lambda i=i,j=j : self.effect_contrast_conditional_params(i,j)))
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
		self.contrasts_interaction = {}
		for i in range(self.k):
			for j in range(i):
				self.contrasts_interaction[(j,i)] = np.kron(self.contrasts[j],self.contrasts[i])

		# indices
		self._tuple_to_design_index = {}
		self._tuple_to_design_index[0] = 0
		ind = 1
		for i in range(self.k):
			for j in range(self.mk[i]):
				self._tuple_to_design_index[(i,j)] = ind
				ind += 1
		for i in range(self.k):
			for j in range(self.mk[i]):
				for k in range(i+1,self.k):
					for l in range(self.mk[k]):
						self._tuple_to_design_index[(i,j,k,l)] = ind
						ind += 1

		self._build_design_matrix()

	def _tuple_to_design_ind(self,tup):
		return self._tuple_to_design_index[tup]

	def _build_design_matrix(self):

		d = 1
		for i in range(self.k):
			d += self.mk[i] - 1
			for j in range(i):
				d += (self.mk[i] - 1) * (self.mk[j] - 1)

		x = np.zeros((self.m,d))
		x[:,0] = 1

		for s in range(self.m):
			ind = 1
			for i in range(self.k):
				x[s,ind:ind+self.mk[i]-1] = self.contrasts[i][self.effect[s,i],:]
				ind += self.mk[i]-1

			for i in range(self.k):
				for j in range(i+1,self.k):
					z = self.effect_contrast_interaction_index(i,self.effect[s,i],j,self.effect[s,j])
					x[s,ind:ind+(self.mk[i]-1)*(self.mk[j]-1)] = self.contrasts_interaction[(i,j)][z,:]

		self._design_matrix = x

	def design_matrix(self):
		return self._design_matrix

	def function_matrix(self,remove=[]):

		if 0 in remove:
			functions = [None]
		else:
			functions = [self.mu_index()]

		for i in range(self.k):
			for j in range(self.mk[i]-1):
				if (i,j) in remove:
					functions.append(None)
				else:
					functions.append(self.effect_contrast_index(i,j))

		for i in range(self.k):
			for k in range(i+1,self.k):
				for j in range(self.mk[i]-1):
					for l in range(self.mk[k]-1):
						if (i,j,k,l) in remove:
							functions.append(None)
						else:
							functions.append(self.effect_contrast_interaction_index(i,j,k,l))

		f = np.zeros((self.n,len(functions)))

		for i,z in enumerate(functions):
			if z is None:
				f[:,i] = 0
			else:
				f[:,i] = self.parameter_cache[z]

		return f

	def residual(self,remove=[]):
		return self.y.T - np.dot(self.design_matrix(),self.function_matrix(remove).T)

	def function_residual(self,f):
		resid = self.residual(remove=[f])

		resid = (resid.T / self.design_matrix()[:,self._tuple_to_design_ind(f)].T).T
		resid = resid[self.design_matrix()[:,self._tuple_to_design_ind(f)]!=0,:]

		return resid

	def offset(self):
		"""offset for the calculation of covariance matrices inverse"""
		return 1e-9

	def effect_index(self,k,l):
		"""lth sample of kth effect"""
		return ['%s_%d(%lf)'%(GP_FANOVA.EFFECT_SUFFIXES[k],l,z) for z in self.x]

	def effect_contrast_index(self,k,l,):
		"""lth sample of kth effect"""
		#return ['%s*_%d(%lf)'%(GP_FANOVA.EFFECT_SUFFIXES[k],l,z) for z in self.x]
		return self._effect_contrast_index[k][l]

	def effect_interaction_index(self,k,l,m,n,contrast=False):
		if k > m:
			k,l,m,n = m,n,k,l # swapperooni

		if contrast:
			return ['(%s:%s)*_(%d,%d)(%lf)'%(GP_FANOVA.EFFECT_SUFFIXES[k],GP_FANOVA.EFFECT_SUFFIXES[m],l,n,z) for z in self.x]
		return ['(%s:%s)_(%d,%d)(%lf)'%(GP_FANOVA.EFFECT_SUFFIXES[k],GP_FANOVA.EFFECT_SUFFIXES[m],l,n,z) for z in self.x]

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

	def effect_contrast_interaction_array(self,i,k,history=None,deriv=False):

		if i > k:
			return self.effect_contrast_array(i,k,history,deriv)

		if deriv:
			loc = self.derivative_history
		elif not history is None:
			loc = self.parameter_history
		else:
			loc = self.parameter_cache

		a = np.zeros((self.n,(self.mk[i]-1)*(self.mk[k]-1)))
		for j in range(self.mk[i]-1):
			for l in range(self.mk[k]-1):
				if history is None:
					a[:,j*self.mk[k]+l] = loc[self.effect_contrast_interaction_index(i,j,k,l)]
				else:
					a[:,j*self.mk[k]+l] = loc.loc[history,self.effect_contrast_interaction_index(i,j,k,l)]
		return a

	def effect_contrast_interaction_index(self,i,j,k,l):
		return j*self.mk[k] + l

	def function_conditional(self,f,kern):
		m = self.function_residual(f)
		n = m.shape[0]
		m = m.mean(0)

		f_inv = kern.K_inv(self.x)
		y_inv = self.y_k.K_inv(self.x)

		A = n*y_inv + f_inv
		b = n*np.dot(y_inv,m)

		chol_A = np.linalg.cholesky(A)
		chol_A_inv = np.linalg.inv(chol_A)
		A_inv = np.dot(chol_A_inv.T,chol_A_inv)

		return np.dot(A_inv,b), A_inv

	# def derivative_conditional(self,f,kern):

	def effect_sample(self,i,j):

		return np.dot(self.effect_contrast_array(i),self.contrasts[i][j,:])

	def y_mu(self):
		return np.dot(self.design_matrix(),self.function_matrix().T).ravel()

	def y_likelihood(self,sigma=None):
		y = np.ravel(self.y.T)
		mu = self.y_mu()
		sigma = pow(10,sigma)

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
			cov = self.effect_contrast_k(i).K(self.x,sigma,lengthscale)
			cov += cov.mean()*np.eye(self.n)*1e-6
			try:
				ll += scipy.stats.multivariate_normal.logpdf(self.parameter_cache[self.effect_contrast_index(i,j)],mu,cov)
			except np.linalg.LinAlgError:
				logger = logging.getLogger(__name__)
				logger.error("likelihood LinAlgError (%d,%d)" % (i,j))

		return ll
