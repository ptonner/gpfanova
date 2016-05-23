from . import Sampler

class Slice(Sampler):
	"""Slice sampling as described in Neal (2003), using the 'step-out' algorithm."""

	def __init__(self,name,parameters,logdensity_fxn,w,m):
		"""Args:
			logdensity_fxn: logarithm of conditional density function of the parameter X that we will evaluate to find our slice interval
			w: an interval step size, defining how large each interval increase is
			m: limits the maximum interval size to w*m

		Returns:
			x1: the new sample of the variable X
			l,r: the final region bounds used
		"""
		Sampler.__init__(self,name,'Slice',parameters,current_param_dependent=True)
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
