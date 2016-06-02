# cython: profile=True

import numpy as np

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
			params[i] = self.model.parameter_cache[self.parameters[i]]

		if self.logspace:
			for i in range(len(params)):
				params[i] = pow(10,params[i])

		return params

	def K(self,X,*args,**kwargs):
		params = self.build_params(*args,**kwargs)
		return self._K(X,*params)

	def K_inv(self,X,*args,**kwargs):
		params = self.build_params(*args,**kwargs)
		K = self._K(X,*params) + np.eye(X.shape[0])*OFFSET
		chol = np.linalg.cholesky(K)
		chol_inv = np.linalg.inv(chol)
		inv = np.dot(chol_inv.T,chol_inv)

		return inv

	def _K(self,X,*args):
		raise NotImplemented()

	def _dK(self,X):
		raise NotImplemented()
