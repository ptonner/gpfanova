from gpfanova.interval import ScalarInterval, FunctionInterval
import gpfanova, os
import numpy as np
import matplotlib.pyplot as plt

class ConvergenceTest(object):

	def __init__(self,label=""):
		self.label=label
		self.intervals = {}
		self.scalarIntervals = []
		self.functionIntervals = []
		self.m = None
		self.trueValues = {}
		self.allChecks = []

	def iterate(self,**kwargs):
		self.initialize(**kwargs)
		self.sample(**kwargs)
		self.checkIntervals()

	def results(self):
		ac = np.array(self.allChecks)

		return ac.sum(0)/ac.shape[0]

	def sample(self,nsample=100,thin=10,**kwargs):
		self.m.sample(nsample,thin=thin)

	def buildModel(self,k=2,r=5,n=50,**kwargs):
		self.x = np.linspace(-1,1,n)[:,None]
		self.y = np.zeros((n,k*r))
		self.effect = np.array(range(k)*r)[:,None]

		self.m = gpfanova.fanova.FANOVA(self.x,self.y,self.effect,**kwargs)
		self.m.y,self.fsamples = self.m.samplePrior()

	def initialize(self,**kwargs):
		self.buildModel(**kwargs)

		for parameter,alpha,args,kwargs in self.scalarIntervals:
			self.trueValues[parameter] = self.m.parameter_cache[parameter]
		for ind,alpha,args,kwargs in self.functionIntervals:
			self.trueValues[ind] = self.fsamples[:,ind]

	def addScalarInterval(self,parameter,alpha,*args,**kwargs):
		self.scalarIntervals.append((parameter,alpha,args,kwargs))

	def addFunctionInterval(self,ind,alpha,*args,**kwargs):
		self.functionIntervals.append((ind,alpha,args,kwargs))

	def checkIntervals(self):
		chex = []
		for parameter,alpha,args,kwargs in self.scalarIntervals:
			ival = ScalarInterval(self.m.parameterSamples(parameter).values[:,0],alpha,*args,**kwargs)
			chex.append(ival.contains(self.trueValues[parameter]))
			self.intervals[parameter] = ival

		for ind,alpha,args,kwargs in self.functionIntervals:
			ival = FunctionInterval(self.m.functionSamples(ind).values,alpha,*args,**kwargs)
			chex.append(ival.contains(self.trueValues[ind]))
			self.intervals[ind] = ival

		self.allChecks.append(chex)

	def plot(self):
		for parameter,_,_,_ in self.scalarIntervals:
			ival = self.intervals[parameter]
			
			plt.figure()
			plt.subplot(121)
			plt.plot(self.m.parameterSamples(parameter))    
			plt.subplot(122)
			ival.plot((-2,2),self.trueValues[parameter])
			plt.savefig(os.path.join('testing',self.label,'%s.pdf'%parameter))

		for ind,_,_,_ in self.functionIntervals:
			ival = self.intervals[ind]
			samples = self.m.functionSamples(ind).values

			plt.figure()
			plt.subplot(121)
			gpfanova.plot.plotFunctionSamples(samples)
			plt.plot(self.fsamples[:,ind],lw=3);

			plt.subplot(122)
			diff = samples - self.fsamples[:,ind]
			cmap = plt.get_cmap("RdBu")
			for i in range(diff.shape[0]):
				plt.plot(abs(diff[i,:]),c=cmap(1.*i/diff.shape[0]));

			plt.savefig(os.path.join('testing',self.label,'f%d.pdf'%ind))


if __name__ == "__main__":

	cvg = ConvergenceTest('test')
	
	cvg.addScalarInterval('prior0_sigma',.05,lambda x: cvg.m.prior_likelihood(0,sigma=x,prior_lb=-2,prior_ub=2))
	cvg.addScalarInterval('prior0_lengthscale',.05,lambda x: cvg.m.prior_likelihood(0,lengthscale=x,prior_lb=-2,prior_ub=2))
	cvg.addScalarInterval('prior1_sigma',.05,lambda x: cvg.m.prior_likelihood(1,sigma=x,prior_lb=-2,prior_ub=2))
	cvg.addScalarInterval('prior1_lengthscale',.05,lambda x: cvg.m.prior_likelihood(1,lengthscale=x,prior_lb=-2,prior_ub=2))

	cvg.addFunctionInterval(0,.95)
	cvg.addFunctionInterval(1,.95)

	cvg.iterate(nsample=1000,n=10,r=3,y_sigma=-2)

	print cvg.results()

	cvg.plot()