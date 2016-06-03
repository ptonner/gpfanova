# cython: profile=True

from patsy.contrasts import Sum
from sample import SamplerContainer, Gibbs, Slice, Fixed, Transform, Function
from kernel import RBF, White
import numpy as np
import GPy, scipy, logging

class Base(SamplerContainer):

	EFFECT_SUFFIXES = ['alpha','beta','gamma','delta','epsilon']

	def __init__(self,x,y):

		self.x = x # independent variables
		self.y = y # dependent variables

		self.n = self.x.shape[0]
		assert self.y.shape[0] == self.n, 'x and y must have same first dimension shape!'
		self.p = self.x.shape[1]
		self.m = self.y.shape[1]

		self.design_matrix = None
		self.build_design_matrix()

		# number of functions being estimated
		self.f = self.design_matrix.shape[1]

		# kernel and sampler
		self.y_k = White(self,['y_sigma'],logspace=True)
		samplers = [Slice('y_sigma','y_sigma',self.y_likelihood,.1,10)]

		# function priors
		self.kernels = []
		for i,p in enumerate(self.prior_groups()):
			self.kernels.append(RBF(self,['prior%d_sigma'%i,'prior%d_lengthscale'%i],logspace=True))

			for f in p:
				samplers.append(Function('f%d'%f,self.function_index(f),self,f,self.kernels[-1]))

			samplers.append(Fixed('prior%d_sigma'%i,'prior%d_sigma'%i))
			samplers.append(Fixed('prior%d_lengthscale'%i,'prior%d_lengthscale'%i))

		SamplerContainer.__init__(self,*samplers)

	def build_design_matrix(self):
		if self.design_matrix is None:
			self.design_matrix = self._build_design_matrix()

	def _build_design_matrix(self):
		raise NotImplemented("Implement a design matrix for your model!")

	def function_index(self,i):
		return ['f%d(%lf)'%(i,z) for z in self.x]

	def prior_groups(self):
		raise NotImplemented("Implement a prior grouping function for your model!")

	def function_matrix(self,remove=[],only=[]):

		functions = []

		if len(only) > 0:
			for o in only:
				functions.append(self.function_index(o))
		else:
			for f in range(self.f):
				if f in remove:
					functions.append(None)
				else:
					functions.append(self.function_index(f))

		f = np.zeros((self.n,len(functions)))

		for i,z in enumerate(functions):
			if z is None:
				f[:,i] = 0
			else:
				f[:,i] = self.parameter_cache[z]

		return f

	def residual(self,remove=[],only=[]):
		return self.y.T - np.dot(self.design_matrix,self.function_matrix(remove,only).T)

	def function_residual(self,f):
		resid = self.residual(remove=[f])

		resid = (resid.T / self.design_matrix[:,f].T).T
		resid = resid[self.design_matrix[:,f]!=0,:]

		return resid

	def offset(self):
		"""offset for the calculation of covariance matrices inverse"""
		return 1e-9

	def y_k_inv(self,x=None):
		if x is None:
			x = self.x

		return self.y_k.K_inv(x)

	def y_mu(self):
		return np.dot(self.design_matrix,self.function_matrix().T).ravel()

	def y_likelihood(self,sigma=None):
		y = np.ravel(self.y.T)
		mu = self.y_mu()
		sigma = pow(10,sigma)

		return np.sum(scipy.stats.norm.logpdf(y-mu,0,sigma))
