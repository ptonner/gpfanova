
class Kernel(object):

	def __init__(self,model,logspace,*args,**kwargs):
		self.model = model
		self.parameters = args
		self.logspace = logspace

	def build_params(self,*args,**kwargs):
		params = [None] * len(self.parameters)

		# set args as params
		for i in range(len(args)):
			params[i] = args[i]

		# look for params in kwargs
		for p in self.parameters:
			if p in kwargs:
				params[self.parameters.index(p)] = kwargs[p]

		# add any missing params from cache
		while None in params:
			i = params.index(None)
			params[i] = self.model.parameter_cache[self.parameters[i]]

		if self.logspace:
			for i in range(len(params)):
				params[i] = 10**params[i]

		return params

	def K(self,X,*args,**kwargs):
		params = self.build_params(*args,**kwargs)
		return self._K(X,*params)

	def _K(self,X,*args):
		raise NotImplemented()

	def _dK(self,X):
		raise NotImplemented()
