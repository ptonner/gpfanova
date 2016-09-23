from . import Sampler

class Transform(Sampler):
	"""not a true 'sampler', but some variable that is a transform of other variables"""

	def __init__(self,name,parameters,transform_fxn):
		Sampler.__init__(self,name,'Transform',parameters,)
		self.transform_fxn = transform_fxn

	def _sample(self,*args,**kwargs):
		return self.transform_fxn()
