from . import Sampler
import pandas as pd
import time,logging

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

	def _sample(self,random=False):

		ind = range(len(self.samplers))

		if random:
			pyrandom.shuffle(ind)

		for i in ind:
			sampler = self.samplers[i]

			args = []
			if sampler.current_param_dependent:
				param = self.parameter_cache[sampler.parameters]

				# just use the parameter value if length one
				if len(sampler.parameters) == 1:
					param = param[0]

				args += [param]

			sample = sampler.sample(*args)
			self.parameter_cache[sampler.parameters] = sample

	def sample(self,n=1,save=0,verbose=False):
		logger = logging.getLogger(__name__)
		start = self.parameter_history.shape[0]
		i = 1

		start_time = iter_time = time.time()
		while self.parameter_history.shape[0] - start < n:
			self._sample()

			if save == 0 or i % save == 0:
				self.store()

				j = self.parameter_history.shape[0] - start
				logger.debug("%d/%d iterations (%.2lf%s) finished in %.2lf minutes" % (j,n,100.*j/n,'%',(time.time()-start_time)/60))
				iter_time = time.time()

			i+=1

		logger.debug("%d samples finished in %.2lf minutes" % (n, (time.time() - start_time)/60))

	def store(self):
		self.parameter_history = self.parameter_history.append(self.parameter_cache,ignore_index=True)
		self.parameter_history.index = range(self.parameter_history.shape[0])

	def __repr__(self):
		s = "\n".join(['SamplerContainer:']+[str(samp) for samp in self.samplers])

		return s
