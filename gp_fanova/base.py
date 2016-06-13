from patsy.contrasts import Sum
from sample import SamplerContainer, Gibbs, Slice, Fixed, Function
from kernel import RBF, White
import numpy as np
import scipy.stats, logging

class Base(SamplerContainer):
	"""Base for constructing functional models of the form $y(t) = X \times b(t)$

	Subclasses must implement the functions _build_design_matrix and prior_groups.
	_build_design_matrix defines the design matrix for the model, such that
	$y_i(t) = X_i b(t)$ for design matrix $X$. prior_groups returns a list of lists,
	where each list defines a grouping of functions who share a GP prior.
	"""

	def __init__(self,x,y,hyperparam_kwargs={},*args,**kwargs):
		""" Construct the base functional model.

		Args:
			x: np.array (n x p), independent variables (not the design matrix!),
				where obesrvations have been made
			y: np.array (n x r), funtion observations
		"""

		self.x = x # independent variables
		self.y = y # dependent variables

		# initialize the base index list
		self._observation_index_base_list = None

		self.n = self.x.shape[0]
		assert self.y.shape[0] == self.n, 'x and y must have same first dimension shape!'
		self.p = self.x.shape[1]
		self.m = self.y.shape[1]

		self.design_matrix = None
		self.build_design_matrix()

		# number of functions being estimated
		self.f = self.design_matrix.shape[1]

		if self.f > np.linalg.matrix_rank(self.design_matrix):
			logger = logging.getLogger(__name__)
			logger.error("design matrix is of rank %d, but there are %d functions!"%(np.linalg.matrix_rank(self.design_matrix),self.f))

		# kernel and sampler
		self.y_k = White(self,['y_sigma'],logspace=True)
		w,m = .1,10
		if 'y_sigma' in hyperparam_kwargs:
			w,m = hyperparam_kwargs['y_sigma']
		elif 'sigma' in hyperparam_kwargs:
			w,m = hyperparam_kwargs['sigma']
		samplers = [Slice('y_sigma','y_sigma',self.y_likelihood,w,m)]

		# function priors
		self.kernels = []
		fxn_names = self.function_names()
		for i,p in enumerate(self.prior_groups()):
			self.kernels.append(RBF(self,['prior%d_sigma'%i,'prior%d_lengthscale'%i],logspace=True))

			for f in p:
				if f in fxn_names:
					s = fxn_names[f]
				else:
					s = "f%d"%f
				samplers.append(Function('%s'%s,self.function_index(f),self,f,self.kernels[-1]))

			w,m = .1,10
			if 'sigma' in hyperparam_kwargs:
				w,m = hyperparam_kwargs['sigma']
			samplers.append(Slice('prior%d_sigma'%i,'prior%d_sigma'%i,lambda x,p=i: self.prior_likelihood(p=p,sigma=x),w,m))

			w,m = .1,10
			if 'lengthscale' in hyperparam_kwargs:
				w,m = hyperparam_kwargs['lengthscale']
			samplers.append(Slice('prior%d_lengthscale'%i,'prior%d_lengthscale'%i,lambda x,p=i: self.prior_likelihood(p=p,lengthscale=x),w,m))
		samplers.extend(self._additional_samplers())

		SamplerContainer.__init__(self,samplers,**kwargs)

	def _additional_samplers(self):
		"""Additional samplers for the model, can be overwritten by subclasses."""
		return []

	def function_names(self):
		"""Function names, can be overwritten by subclasses.

		returns:
			dict(index:name), keys indicate function index in the
				design matrix, with values representing the name to use for the
				function."""
		return {}

	def build_design_matrix(self):
		if self.design_matrix is None:
			self.design_matrix = self._build_design_matrix()

	def _build_design_matrix(self):
		"""Build a design matrix defining the relation between observations and underlying functions.

		The returned matrix should be shape (n,f), where n is the number observation points,
		and f is the number of functions to be estimated. f will be infered by the
		shape of the matrix returned from this function.
		"""
		raise NotImplementedError("Implement a design matrix for your model!")

	def function_index(self,i):
		"""return the parameter_cache indices for function i"""
		return ['f%d(%s)'%(i,z) for z in self._observation_index_base()]

	def function_prior(self,f):
		"""return the prior index for function f."""

		priors = self.prior_groups()
		for i in range(len(priors)):
			if f in priors[i]:
				return i
		return -1

	def _observation_index_base(self):
		"""return the base indice structure from the observations.

		returns:
			list of strings
		"""
		if self._observation_index_base_list is None:
			self._observation_index_base_list = ['%s'%str(z) for z in self.x]

		return self._observation_index_base_list

	def prior_groups(self):
		raise NotImplementedError("Implement a prior grouping function for your model!")

	def _prior_parameters(self,i):
		if i < 0:
			return ['y_sigma']
		if i >= len(self.prior_groups()):
			return [None]

		return ['prior%d_sigma'%i,'prior%d_lengthscale'%i]

	def function_matrix(self,remove=[],only=[]):
		"""return the current function values, stored in the parameter_cache."""

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
		"""compute the residual Y-Mb, without the function f."""
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
		"""Compute the likelihood of the observations y given the design matrix and latent functions"""
		y = np.ravel(self.y.T)
		mu = self.y_mu()
		sigma = pow(10,sigma)

		return np.sum(scipy.stats.norm.logpdf(y-mu,0,sigma))

	def prior_likelihood(self,p,sigma=None,lengthscale=None):
		"""Compute the likelihood of functions with prior p, for the current/provided hyperparameters"""
		ind = self.prior_groups()[p]

		mu = np.zeros(self.n)
		cov = self.kernels[p].K(self.x,sigma,lengthscale)
		cov += cov.mean()*np.eye(self.n)*1e-6

		rv = scipy.stats.multivariate_normal(mu,cov)

		ll = 1
		for f in ind:
			try:
				ll += rv.logpdf(self.parameter_cache[self.function_index(f)])
			except np.linalg.LinAlgError:
				logger = logging.getLogger(__name__)
				logger.error("prior likelihood LinAlgError (%d,%d)" % (p,f))
		return ll

	def sample_prior(self,):

		# sample the hyperparameters

		## sample the latent functions
		samples = np.zeros((self.f,self.n))
		for i in range(self.f):
			mu = np.zeros(self.n)
			cov = self.kernels[self.function_prior(i)].K(self.x)
			samples[i,:] = scipy.stats.multivariate_normal.rvs(mu,cov)

		## put into data
		y = np.dot(self.design_matrix,samples) + np.random.normal(0,np.sqrt(pow(10,self.parameter_cache['y_sigma'])),size=(self.m,self.n))

		return y.T,samples.T
