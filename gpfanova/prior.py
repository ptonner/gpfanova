import base, linalg, scipy
from kernel import RBF, Kernel
from sample import Slice, Function, FunctionDerivative
import numpy as np

class Prior(object):

	def __init__(self,functions,name,base,p,derivatives,f,n,x,kernel=None):
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

		self.base = base
		self.p = p
		self.derivatives = derivatives
		self.f = f
		self.n = n
		self.x = x
		self.name = name

		self.kernel = self.buildKernel()
		self._samplers = self.buildSamplers()

	def buildKernel(self):

		if self._kernelType is None:
			return RBF(self.base,['prior%s_sigma'%self.name]+['prior%s_lengthscale%d'%(self.name,d) for d in range(self.p)],logspace=True)
		return self._kernelType(self.base,['prior%s_sigma'%self.name]+['prior%s_lengthscale%d'%(self.name,d) for d in range(self.p)],logspace=True)

	def samplers(self):
		return self._samplers

	def buildSamplers(self,):

		samplers = []
		for f in self._functions:
			samplers.append(Function('%d'%f,self.x,self.base,f,self.kernel))

			if self.derivatives:
				for d in range(self.p):
					samplers.append(FunctionDerivative('d%f'%f,d,self.base.functionIndex(f,derivative=True),self.base,f,self.kernel))

		samplers.extend(self.kernelSamplers())
		return samplers

	def kernelSamplers(self):

		w,m = .1,10
		samplers = []
		for p in self.kernel.parameters:
			samplers.append(Slice(p,p,lambda x: self.loglikelihood(**{p:x}),w,m))

		return samplers

	def functions(self):
		return self._functions

	def sample(self):
		"""Sample functions from this prior."""
		samples = np.zeros((self.f,self.n))

		for f in self._functions:

			mu = np.zeros(self.n)
			cov = self.kernel.K(self.x)
			L = linalg.jitchol(cov)
			cov = np.dot(L,L.T)

			samples[f,:] = scipy.stats.multivariate_normal.rvs(mu,cov)

		return samples

	def loglikelihood(self,*args,**kwargs):
		mu = np.zeros(self.n)
		cov = self.kernel.K(self.x,*args,**kwargs)

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
