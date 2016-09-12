# cython: profile=True

import numpy as np
import logging
from .. import linalg

OFFSET = 1e-9

class Kernel(object):

	def __init__(self,model,params,logspace,*args,**kwargs):
		self.model = model
		self.parameters = params
		self.logspace = logspace

	def build_params(self,*args,**kwargs):
		params = [None] * len(self.parameters)

		# set args as params
		for i in range(len(args)):
			if args[i] is None:
				continue
			params[i] = args[i]

		# look for params in kwargs
		for p in self.parameters:
			if p in kwargs and not kwargs[p] is None:
				params[self.parameters.index(p)] = kwargs[p]

		# add any missing params from cache
		while None in params:
			i = params.index(None)
			params[i] = self.model.get(self.parameters[i])

		if self.logspace:
			for i in range(len(params)):
				params[i] = pow(10,params[i])

		return params

	def K(self,X,*args,**kwargs):
		"""Compute the covariance matrix for the input X with alternative values of hyperparameters provided through args and kwargs, if necessary."""
		params = self.build_params(*args,**kwargs)
		return self._K(X,*params)

	def dK(self,X,cross=False,*args,**kwargs):
		"""Compute the derivative of the covariance matrix for the input X with alternative values of hyperparameters provided through args and kwargs, if necessary."""
		params = self.build_params(*args,**kwargs)
		return self._dK(X,cross,*params)

	def K_inv(self,X,*args,**kwargs):
		params = self.build_params(*args,**kwargs)
		K = self._K(X,*params) + np.eye(X.shape[0])*OFFSET
		try:
			# chol = np.linalg.cholesky(K)
			chol = linalg.jitchol(K)
			chol_inv = np.linalg.inv(chol)
		except np.linalg.linalg.LinAlgError,e:
			logger = logging.getLogger(__name__)
			logger.error('Kernel inversion error: %s'%str(self.parameters))
			raise(e)
		inv = np.dot(chol_inv.T,chol_inv)

		return inv

	def _K(self,X,*args):
		raise NotImplemented()

	def _dK(self,X,cross,*args):
		raise NotImplemented()
