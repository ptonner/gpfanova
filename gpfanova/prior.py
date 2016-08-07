import base, linalg, scipy
from kernel import RBF
from sample import Slice, Function, FunctionDerivative
import numpy as np

class Prior(object):

	def __init__(self,functions,name=None):
		self._functions = functions
		self.base = None
		self.kernel = None
		self.name = name

	def buildKernel(self):

		self.kernel = RBF(self.base,['prior%s_sigma'%self.name]+['prior%s_lengthscale%d'%(self.name,d) for d in range(self.base.p)],logspace=True)

	def samplers(self,):
		if self.base is None:
			raise AttributeError("this prior has not been initialized with its base class!")

		self.samplers = []
		for f in self._functions:
			self.samplers.append(Function('%d'%f,self.base.functionIndex(f),self.base,f,self.kernel))

			if self.base.derivatives:
				for d in range(self.base.p):
					self.samplers.append(FunctionDerivative('d%f'%f,d,self.base.functionIndex(f,derivative=True),self.base,f,self.kernel))

		self.samplers.extend(self.kernelSamplers())
		return self.samplers

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

		self.buildKernel()

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

		# use cholesky jitter code to find PD covariance matrix
		# diagA = np.diag(cov)
		# if np.any(diagA <= 0.):
		# 	from scipy import linalg
		# 	raise linalg.LinAlgError("not pd: non-positive diagonal elements")
		# jitter = diagA.mean() * 1e-6
		# num_tries = 1
		# maxtries=10
		# rv = None
		# while num_tries <= maxtries and np.isfinite(jitter):
		# 	print 'jitter'
		# 	try:
		# 		rv = scipy.stats.multivariate_normal(mu,cov + np.eye(cov.shape[0]) * jitter)
		# 		break
		# 	except:
		# 		jitter *= 10
		# 	finally:
		# 		num_tries += 1

		# try:
		# 	rv = scipy.stats.multivariate_normal(mu,cov)
		# except np.linalg.LinAlgError,e:
		# 	print args
		# 	print kwargs
		# 	raise e
		rv = scipy.stats.multivariate_normal(mu,cov)

		ll = 0
		for f in self._functions:
			try:
				ll += rv.logpdf(self.base.get(self.base.functionIndex(f)))
			except np.linalg.LinAlgError:
				logger = logging.getLogger(__name__)
				logger.error("prior likelihood LinAlgError (%d,%d)" % (p,f))

		return ll
