from . import Sampler

class Fixed(Sampler):
	"""dummy class for fixed parameters, used as placeholder until sampler implemented"""

	def __init__(self,name,parameters,):
		Sampler.__init__(self,name,'Fixed',parameters,current_param_dependent=True)

	def _sample(self,x,*args,**kwargs):
		return x
