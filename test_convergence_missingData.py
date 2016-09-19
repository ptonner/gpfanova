from test_convergence import ConvergenceTest
import numpy as np

class ConvergenceTestMissingData(ConvergenceTest):

    def buildModel(self,missing,**kwargs):
        ConvergenceTest.buildModel(self,**kwargs)

        missing = np.array([m in missing for m in range(self.y.shape[0])])

        self.ymissing = self.m.y.copy()
        self.ymissing[~missing,:] = np.nan
        self.m.y[missing,:] = np.nan

if __name__ == "__main__":

    cvg = ConvergenceTestMissingData('test-singleEffect-2k-missingData')
    cvg.addScalarInterval('y_sigma',.05,lambda x: cvg.m.observationLikelihood(sigma=x,prior_lb=-2,prior_ub=2))
    cvg.addScalarInterval('prior0_sigma',.05,lambda x: cvg.m.prior_likelihood(0,sigma=x,prior_lb=-2,prior_ub=2))
    cvg.addScalarInterval('prior0_lengthscale',.05,lambda x: cvg.m.prior_likelihood(0,lengthscale=x,prior_lb=-2,prior_ub=2))
    cvg.addScalarInterval('prior1_sigma',.05,lambda x: cvg.m.prior_likelihood(1,sigma=x,prior_lb=-2,prior_ub=2))
    cvg.addScalarInterval('prior1_lengthscale',.05,lambda x: cvg.m.prior_likelihood(1,lengthscale=x,prior_lb=-2,prior_ub=2))

    cvg.addFunctionInterval(0,.95)
    cvg.addFunctionInterval(1,.95)

    def permute(m):
        #m.parameter_cache['y_sigma'] = np.random.uniform(-1,1)

        m.parameter_cache['prior0_sigma'] = np.random.uniform(-1,1)
        m.parameter_cache['prior0_lengthscale'] = np.random.uniform(-1,1)

        m.parameter_cache['prior1_sigma'] = np.random.uniform(-1,1)
        m.parameter_cache['prior1_lengthscale'] = np.random.uniform(-1,1)

        return m

    for i in range(10):
        if i % 10 == 0:
            print i
        cvg.iterate(nsample=10000,thin=10,missing=[2,7],n=10,r=2,y_sigma=-2,plot=True,save=True,permutationFunction=permute,burnin=200)

    cvg.save()
