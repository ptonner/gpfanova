import base, linalg, scipy
from kernel import RBF, Kernel
from sample import Slice, Function, FunctionDerivative
import numpy as np

class Prior(object):

	def __init__(self,functions,name=None,kernel=None):
		self._functions = functions
		assert issubclass(type(self._functions),list), 'must provide a list of functions!'

		try:
			self._functions = [int(f) for f in self._functions]
		except:
			raise ValueError("must provide list of intergers for functions!")

		self._kernelType = kernel
		if not self._kernelType is None:
			if not issubclass(self._kernelType,Kernel):
				raise ValueError("Must provide valid kernel type, %s provided." % (str(self._kernelType)))

		self.base = None
		self.kernel = self.buildKernel
		self.name = name
		self._samplers = None

	def buildKernel(self):

		return RBF(self.base,['prior%s_sigma'%self.name]+['prior%s_lengthscale%d'%(self.name,d) for d in range(self.base.p)],logspace=True)

	def samplers(self):
		return self._samplers

	def buildSamplers(self,):
		if self.base is None:
			raise AttributeError("this prior has not been initialized with its base class!")

		samplers = []
		for f in self._functions:
			samplers.append(Function('%d'%f,self.base.functionIndex(f),self.base,f,self.kernel))

			if self.base.derivatives:
				for d in range(self.base.p):
					samplers.append(FunctionDerivative('d%f'%f,d,self.base.functionIndex(f,derivative=True),self.base,f,self.kernel))

		samplers.extend(self.kernelSamplers())
		return samplers

	def kernelSamplers(self):

		w,m = .1,10
		samplers = []
		for p in self.kernel.parameters:
			samplers.append(Slice(p,p,lambda x: self.loglikelihood(**{p:x}),w,m))

		return samplers

	def setBase(self,b):
		self.base = b

		if not isinstance(self.base,base.Base):
			raise TypeError("Must provide an instance of base.Base!")

		self.kernel = self.buildKernel()
		self._samplers = self.buildSamplers()

	def functions(self):
		return self._functions

	def sample(self):
		"""Sample functions from this prior."""
		samples = np.zeros((self.base.f,self.base.n))

		for f in self._functions:

			mu = np.zeros(self.base.n)
			cov = self.kernel.K(self.base.x)
			L = linalg.jitchol(cov)
			cov = np.dot(L,L.T)

			samples[f,:] = scipy.stats.multivariate_normal.rvs(mu,cov)

		return samples

	def loglikelihood(self,*args,**kwargs):
		mu = np.zeros(self.base.n)
		cov = self.kernel.K(self.base.x,*args,**kwargs)

		L = linalg.jitchol(cov)
		cov = np.dot(L,L.T)

		rv = scipy.stats.multivariate_normal(mu,cov)

		ll = 0
		for f in self._functions:
			try:
				ll += rv.logpdf(self.base.get(self.base.functionIndex(f)))
			except np.linalg.LinAlgError:
				logger = logging.getLogger(__name__)
				logger.error("prior likelihood LinAlgError (%d,%d)" % (p,f))

		return ll

def PriorList(object):

	def __init__(self,*args):

		self._priors = []
		for p in args:
			if issubclass(type(p),Prior):
				self._priors.append(p)

	def __getitem__(self,ind):
		return self._priors[ind]
