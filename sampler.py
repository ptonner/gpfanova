import random as pyrandom
import pandas as pd
import scipy

class Sampler(object):

	def __init__(self,name,_type,parameters,current_param_dependent=False,*args,**kwagrs):
		"""
		Args:
			current_param_dependent: does the _sample function depend on the current parameter estimate? e.g. slice sampling, MH
		"""
		self.name = name
		self.type = _type
		self.current_param_dependent = current_param_dependent

		if type(parameters) == str:
			self.parameters = [parameters]
		else:
			self.parameters = parameters

	def sample(self,*args,**kwargs):
		return self._sample(*args,**kwargs)

	def _sample(self,*args,**kwargs):
		raise NotImplemented("implement this function for your sampler!")

	def __repr__(self):
		if len(self.parameters) < 3:
			return "%s (%s): %s" % (self.name, self.type, ', '.join(self.parameters))
		return "%s (%s): %s, ..." % (self.name, self.type, ', '.join(self.parameters[:3]))

class SamplerContainer(object):

	def __init__(self,*args):
		self.samplers = [a for a in args if issubclass(type(a),Sampler)]

		ind = self.build_index()
		self.parameter_cache = pd.Series([0]*len(ind),index=ind)
		self.parameter_history = pd.DataFrame(columns=ind)

	def build_index(self):
		ind = []
		for s in self.samplers:
			ind += s.parameters

		return ind

	# def _sample(self,random=True):
	#
	# 	ind = len(self.samplers)
	#
	# 	if random:
	# 		pyrandom.shuffle(ind)
	#
	# 	for i in ind:
	# 		sampler = self.samplers[i]
	#
	# 		sample = sampler.sample(*args)
	# 		self.parameter_cache[sampler.parameters] = sample

	def sample(self,random=False):

		ind = range(len(self.samplers))

		if random:
			pyrandom.shuffle(ind)

		for i in ind:
			sampler = self.samplers[i]

			args = []
			if sampler.current_param_dependent:
				args += [self.parameter_cache[sampler.parameters]]

			print sampler

			sample = sampler.sample(*args)
			self.parameter_cache[sampler.parameters] = sample


	def __repr__(self):
		s = "\n".join(['SamplerContainer:']+[str(samp) for samp in self.samplers])

		return s


class Fixed(Sampler):
	"""dummy class for fixed parameters, used as placeholder until sampler implemented"""

	def __init__(self,name,parameters,):
		Sampler.__init__(self,name,'Fixed',parameters,current_param_dependent=True)

	def _sample(self,x,*args,**kwargs):
		return x


class Gibbs(Sampler):

	def __init__(self,name,parameters,conditional_parameters_fxn,):
		Sampler.__init__(self,name,'Gibbs',parameters)
		self.conditional_parameters_fxn = conditional_parameters_fxn

	def _sample(self,*args,**kwargs):

		mu,cov = self.conditional_parameters_fxn(*args,**kwargs)
		sample = scipy.stats.multivariate_normal.rvs(mu,cov)
		return sample

class Slice(Sampler):
	"""Slice sampling as described in Neal (2003), using the 'step-out' algorithm."""

	def _init__(self,name,parameters,logdensity_fxn,w,m):
		"""Args:
			logdensity_fxn: logarithm of conditional density function of the parameter X that we will evaluate to find our slice interval
			w: an interval step size, defining how large each interval increase is
			m: limits the maximum interval size to w*m

		Returns:
			x1: the new sample of the variable X
			l,r: the final region bounds used
		"""
		Sampler.__init__(self,name,'Slice',parameters)
		self.logdensity_fxn = logdensity_fxn
		self.w = w
		self.m = m

	def _sample(self,x):

		f0 = self.logdensity_fxn(x)
		z = f0 - scipy.stats.expon.rvs(1)

		# find our interval
		u = scipy.stats.uniform.rvs(0,1)
		l = x-self.w*u
		r = l+self.w

		v = scipy.stats.uniform.rvs(0,1)
		j = int(self.m*v)
		k = self.m-1-j

		while j > 0 and z < self.logdensity_fxn(l):
			j -= 1
			l -= self.w

		while k > 0 and z < self.logdensity_fxn(r):
			k -= 1
			r += self.w

		# pick a new point
		u = scipy.stats.uniform.rvs(0,1)
		x1 = l + u*(r-l)

		while z > self.logdensity_fxn(x1):
			if x1 < x:
				l = x1
			else:
				r = x1

			u = scipy.stats.uniform.rvs(0,1)
			x1 = l + u*(r-l)

		return x1
