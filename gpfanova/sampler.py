from distribution import Distribution
from parameter import Parameter
import linalg, scipy
import numpy as np
from collections import OrderedDict

class Sampler(object):

    def __init__(self,parameter,prior,likelihoods):
        assert issubclass(type(parameter),Parameter),'must provide Parameter for parameter, %s given.' % type(parameter)
        assert issubclass(type(prior),Distribution),'must provide Distribution for prior, %s given.' % type(prior)

        self.parameter = parameter
        self.prior = prior
        self.likelihoods = [a for a in likelihoods if issubclass(type(a),Distribution)]

    def sample(self,*args,**kwargs):
        self.parameter.value = self._sample(*args,**kwargs)

class FunctionSampler(Sampler):

    def __init__(self,parameter,prior,likelihoods,x):
        Sampler.__init__(self,parameter,prior,likelihoods)
        self.x = x

        self.indices = []
        for l in self.likelihoods:
            self.indices.append(l.parameters.parameters.keys().index(self.parameter.name))

    def residual(self,):
        n = []
        resid = []
        for i,l in zip(self.indices,self.likelihoods):
            n.append(l.beta[i])
            resid.append(l.residual(remove=[i])/l.beta[i])

        n = np.power(n,2)
        resid = np.column_stack(tuple(resid))*n

        return resid,n

    def buildParameters(self):
        m,n = self.residual()

		# missingValues = np.isnan(m)
		# m = np.nansum((m.T*n).T,0)
		# n = np.sum(((~missingValues).T*n).T,0)

		# y_inv = self.likelihood.kernel.K_inv(self.x)
        y_inv = [l.kernel.K_inv(self.x) for l in self.likelihoods]
        f_inv = self.prior.kernel.K_inv(self.x)

        A = sum([z*yi for z,yi in zip(n,y_inv)]) + f_inv
        b = sum([np.dot(yi,m[:,i]) for i,yi in zip(range(m.shape[1]),y_inv)])

        chol_A = linalg.jitchol(A)
        chol_A_inv = np.linalg.inv(chol_A)
        A_inv = np.dot(chol_A_inv.T,chol_A_inv)

        mu,cov = np.dot(A_inv,b), A_inv

        return mu,cov,b,A_inv

    def _sample(self,*args,**kwargs):
        mu,cov,_,_ = self.buildParameters()

        return scipy.stats.multivariate_normal.rvs(mu,cov)

class MCMC(object):

    def __init__(self,*samplers):
        self.samplers = [a for a in samplers if issubclass(type(a),Sampler)]
        self.parameters = [s.parameter for s in self.samplers]
        self.iterations = []

    def freeze(self,):
        iteration = OrderedDict()
        for p in self.parameters:
            iteration[p.name] = p.value
        self.iterations.append(iteration)

    def sample(self,):
        for s in self.samplers:
            s.sample()
        self.freeze()
