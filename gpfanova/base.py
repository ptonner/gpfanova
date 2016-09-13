from patsy.contrasts import Sum
from sample import SamplerContainer, Slice, Fixed, Function, FunctionDerivative
from kernel import RBF, White
import numpy as np
import scipy.stats, logging
from prior import Prior

class Base(SamplerContainer):
	"""Base for constructing functional models of the form $y(t) = X \times b(t)$

	Subclasses must implement the functions _buildDesignMatrix and priorGroups.
	_buildDesignMatrix defines the design matrix for the model, such that
	$y_i(t) = X_i b(t)$ for design matrix $X$. priorGroups returns a list of lists,
	where each list defines a grouping of functions who share a GP prior.
	"""

	def __init__(self,x,y,designMatrix=None,priors=None,derivatives=False,*args,**kwargs):
		self.x = x
		self.y = y

		self.n = self.x.shape[0]
		assert self.y.shape[0] == self.n, 'x and y must have same first dimension shape!'

		if self.x.ndim > 1:
			self.p = self.x.shape[1]
		else:
			self.p = 1
		self.m = self.y.shape[1]

		self.derivatives = derivatives

		# this list holds the strings representing the values where
		# functions are to be sampled, e.g. x_1, x_2, ...
		self._observationIndexBaseList = ['%s'%str(z) for z in self.x]

		self.designMatrix = designMatrix
		self.checkDesignMatrix()

		if priors is None:
			priors = [(range(self.f),'mean')]
		self.priors = self.buildPriors(priors)
		self.checkPriors()
		self.k = len(priors)

		self.y_k = White(self,['ySigma'],logspace=True)
		w,m = .1,10
		samplers = [Slice('ySigma','ySigma',self.observationLikelihood,w,m)]

		for p in self.priors:
			samplers.extend(p.samplers())

		samplers.extend(self._additionalSamplers())

		SamplerContainer.__init__(self,samplers,**kwargs)

	def checkDesignMatrix(self,):
		# use built-in design matrix contstruction if none provided
		if self.designMatrix is None:
			self.designMatrix = self._buildDesignMatrix()
		# self.buildDesignMatrix()

		self.f = self.designMatrix.shape[1]

		if not self.designMatrix.shape[0] == self.y.shape[1]:
			raise ValueError('Shape mismatch: designMatrix (%d) and y (%d)!' %(self.designMatrix.shape[0],self.y.shape[1]))

		if self.f > np.linalg.matrix_rank(self.designMatrix):
			raise ValueError("design matrix is of rank %d, but there are %d functions!"
							%(np.linalg.matrix_rank(self.designMatrix),self.f))

	def buildPriors(self,priors,kernel=None):
		ret = []

		for fxns,name in priors:
			ret.append(Prior(fxns,name,\
								self,self.p,self.derivatives,self.f,\
								self.n,self.x,kernel=kernel))

		return ret

	def checkPriors(self):
		# for p in self.priors:
		# 	p.setBase(self)

		coverage = [False]*self.f

		for p in self.priors:
			for f in p.functions():
				if coverage[f]:
					raise ValueError("multiple priors supplied for function %d!"%f)
				coverage[f] = True

		if not all(coverage):
			missing = np.where(~np.array(coverage))[0]
			raise ValueError("Missing prior for functions %s!"%",".join(missing))

	def _additionalSamplers(self):
		"""Additional samplers for the model, can be overwritten by subclasses."""
		return []

	def functionNames(self):
		"""Function names, can be overwritten by subclasses.

		returns:
			dict(index:name), keys indicate function index in the
				design matrix, with values representing the name to use for the
				function."""
		return {}

	# def buildDesignMatrix(self):
	# 	if self.designMatrix is None:
	# 		self.designMatrix = self._buildDesignMatrix()

	def _buildDesignMatrix(self):
		"""Build a design matrix defining the relation between observations and underlying functions.

		The returned matrix should be shape (n,f), where n is the number observation points,
		and f is the number of functions to be estimated. f will be infered by the
		shape of the matrix returned from this function.
		"""
		raise NotImplementedError("Implement a design matrix for your model!")

	def functionIndex(self,i,derivative=False,*args,**kwargs):
		"""return the parameter_cache indices for function i"""
		if derivative:
			return ['df%d(%s)'%(i,z) for z in self._observationIndexBase()]
		return ['f%d(%s)'%(i,z) for z in self._observationIndexBase()]

	def functionPrior(self,f):
		"""return the prior index for function f."""

		priors = self.priorGroups()
		for i in range(len(priors)):
			if f in priors[i]:
				return i
		return -1

	def _observationIndexBase(self):
		"""return the base indice structure from the observations.

		returns:
			list of strings
		"""
		if self._observationIndexBaseList is None:
			self._observationIndexBaseList = ['%s'%str(z) for z in self.x]

		return self._observationIndexBaseList

	def priorGroups(self):
		raise NotImplementedError("Implement a prior grouping function for your model!")

	def _priorParameters(self,i):
		if i < 0:
			return ['ySigma']
		if i >= len(self.priorGroups()):
			return [None]

		return ['prior%d_sigma'%i]+['prior%d_lengthscale%d'%(i,d) for d in range(self.p)]

	def functionMatrix(self,remove=[],only=[],derivative=False):
		"""return the current function values, stored in the parameter_cache."""

		functions = []

		if len(only) > 0:
			for o in only:
				functions.append(self.functionIndex(o,derivative=derivative))
		else:
			for f in range(self.f):
				if f in remove:
					functions.append(None)
				else:
					functions.append(self.functionIndex(f,derivative=derivative))

		f = np.zeros((self.n,len(functions)))

		for i,z in enumerate(functions):
			if z is None:
				f[:,i] = 0
			else:
				f[:,i] = self.parameter_cache[z]

		return f

	def functionSamples(self,f,*args,**kwargs):
		"""return the samples of function f from the parameter history."""
		return self.parameter_history[self.functionIndex(f,*args,**kwargs)]

	def residual(self,remove=[],only=[]):
		return self.y.T - np.dot(self.designMatrix,self.functionMatrix(remove,only).T)

	def functionResidual(self,f):
		"""compute the residual Y-Mb, without the function f."""
		resid = self.residual(remove=[f])

		resid = (resid.T / self.designMatrix[:,f].T).T
		resid = resid[self.designMatrix[:,f]!=0,:]

		return resid

	def offset(self):
		"""offset for the calculation of covariance matrices inverse"""
		return 1e-9

	def observationMean(self):
		"""The *conditional* mean of the observations given all functions"""
		return np.dot(self.designMatrix,self.functionMatrix().T).ravel()

	def observationLikelihood(self,sigma=None):
		"""Compute the conditional likelihood of the observations y given the design matrix and latent functions"""
		y = np.ravel(self.y.T)
		mu = self.observationMean()
		sigma = pow(10,sigma)
		sigma = pow(sigma,.5)

		# remove missing values
		mu = mu[~np.isnan(y)]
		y = y[~np.isnan(y)]

		return np.sum(scipy.stats.norm.logpdf(y-mu,0,sigma))

	def samplePrior(self,):

		# sample the hyperparameters

		## sample the latent functions
		samples = np.zeros((self.f,self.n))
		for p in self.priors:
			samples += p.sample()

		## put into data
		y = np.dot(self.designMatrix,samples) + np.random.normal(0,np.sqrt(pow(10,self.parameter_cache['ySigma'])),size=(self.m,self.n))

		return y.T,samples.T
