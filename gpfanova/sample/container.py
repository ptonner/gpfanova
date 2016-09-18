from . import Sampler
import pandas as pd
import time,logging
import random as pyrandom

class SamplerContainer(object):

	def __init__(self,samplers,parameterFile=None,*args,**kwargs):
		self.samplers = [a for a in samplers if issubclass(type(a),Sampler)]
		self.sampler_dict = {a.name:a for a in self.samplers}

		ind = self.build_index()
		self.parameter_cache = pd.Series([0.0]*len(ind),index=ind)
		self.parameter_history = pd.DataFrame(columns=ind)

		for k,v in kwargs.iteritems():
			if k in self.sampler_dict:
				self.parameter_cache[self.sampler_dict[k].parameters] = v

		self.load(parameterFile)

	def build_index(self):
		ind = []
		for s in self.samplers:
			ind += s.parameters

		return ind

	def _sample(self,random=False):

		logger = logging.getLogger(__name__)
		ind = range(len(self.samplers))

		if random:
			pyrandom.shuffle(ind)

		for i in ind:
			sampler = self.samplers[i]

			logger.debug('sampling %s'%str(sampler))

			args = []
			if sampler.current_param_dependent:
				param = self.parameter_cache[sampler.parameters]

				# just use the parameter value if length one
				if len(sampler.parameters) == 1:
					param = param[0]

				args += [param]

			sample = sampler.sample(*args)
			self.parameter_cache[sampler.parameters] = sample

	def sample(self,n=1,thin=0,verbose=False,random=False):
		logger = logging.getLogger(__name__)
		start = self.parameter_history.shape[0]
		i = 1

		start_time = iter_time = time.time()
		# while self.parameter_history.shape[0] - start < n:
		for i in range(n):
			self._sample(random=random)

			# add one so the first save is on the 'thin'th iteration, not the first
			if thin == 0 or (i+1) % thin == 0:
				self.store()

				j = self.parameter_history.shape[0] - start

				if verbose:
					logger.info("%d/%d iterations (%.2lf%s) finished in %.2lf minutes" % (j,n,100.*j/n,'%',(time.time()-start_time)/60))
				iter_time = time.time()

			# i+=1

		if verbose:
			logger.info("%d samples finished in %.2lf minutes" % (n, (time.time() - start_time)/60))

	def parameterSamples(self,name):
		if not name in self.sampler_dict:
			return None
		return self.parameter_history[self.sampler_dict[name].parameters]

	def parameterCurrent(self,name):
		if not name in self.sampler_dict:
			return None
		return self.parameter_cache[self.sampler_dict[name].parameters]

	def store(self):
		self.parameter_history = self.parameter_history.append(self.parameter_cache,ignore_index=True)
		self.parameter_history.index = range(self.parameter_history.shape[0])

	def save(self,f):
		self.parameter_history.to_csv(f,index=False)

	def load(self,f):
		if f is None:
			return
		self.parameter_history = pd.read_csv(f,index_col=None)

	def __repr__(self):
		s = "\n".join(['SamplerContainer:']+[str(samp) for samp in self.samplers])

		return s
