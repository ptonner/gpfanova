from test_convergence import ConvergenceTest
import numpy as np
import itertools, gpfanova

class ConvergenceTestReplicate(ConvergenceTest):

    def buildModel(self,r=5,n=50,**kwargs):
        self.x = np.linspace(-10,10,n)[:,None]
        self.y = np.zeros((n,r))

        dm = np.ones((r,1))

        if self.other is None:
            self.m = gpfanova.base.Base_withReplicate(self.x,self.y,designMatrix=dm,priorGroups=[[0]],**kwargs)
        else:
            self.m = gpfanova.base.Base(self.x,self.y,designMatrix=dm,priorGroups=[[0]],**kwargs)

        self.m.parameter_cache.y_sigma = -1
        self.m.parameter_cache.replicate_lengthscale = 2;
        self.m.parameter_cache.prior0_lengthscale=.2

        if self.other is None:
            self.m.y,self.fsamples = self.m.samplePrior()
        else:
            self.m.y = self.other.m.y
            self.fsamples = self.other.fsamples

    def __init__(self,other=None,*args,**kwargs):
        ConvergenceTest.__init__(self,*args,**kwargs)
        self.other = other

if __name__ == "__main__":

    import argparse, time, os
    parser = argparse.ArgumentParser(description='Run convergence test for Base model with replicate GP prior.')
    parser.add_argument('-r', dest='r', action='store',default=20,
                       help='number of replicates')

    args = parser.parse_args()

    odir = 'test-replicate-%s'%time.ctime().replace(" ","-")
    os.mkdir(os.path.join('testing',odir))

    cvg = ConvergenceTestReplicate(None,odir+"/withReplicate")
    cvg.addScalarInterval('y_sigma',.05,lambda x: cvg.m.observationLikelihood(sigma=x,prior_lb=-2,prior_ub=2))
    cvg.addScalarInterval('prior0_sigma',.05,lambda x: cvg.m.prior_likelihood(0,sigma=x,prior_lb=-2,prior_ub=2))
    cvg.addScalarInterval('prior0_lengthscale',.05,lambda x: cvg.m.prior_likelihood(0,lengthscale=x,prior_lb=-2,prior_ub=2))

    # cvg2 = ConvergenceTestReplicate(cvg,'test-replicate-%s'%time.ctime().replace(" ","-"))
    cvg2 = ConvergenceTestReplicate(cvg,odir+"/withoutReplicate")
    cvg2.addScalarInterval('y_sigma',.05,lambda x: cvg.m.observationLikelihood(sigma=x,prior_lb=-2,prior_ub=2))
    cvg2.addScalarInterval('prior0_sigma',.05,lambda x: cvg.m.prior_likelihood(0,sigma=x,prior_lb=-2,prior_ub=2))
    cvg2.addScalarInterval('prior0_lengthscale',.05,lambda x: cvg.m.prior_likelihood(0,lengthscale=x,prior_lb=-2,prior_ub=2))

    def permute(m):
        #m.parameter_cache['y_sigma'] = np.random.uniform(-2,0)

        # m.parameter_cache['prior0_sigma'] = np.random.uniform(-1,1)
        # m.parameter_cache['prior0_lengthscale'] = np.random.uniform(-1,1)

        return m

    cvg.buildModel()
    for f in range(cvg.m.f):
        cvg.addFunctionInterval(f,.95)
        cvg2.addFunctionInterval(f,.95)

    for i in range(50):
        if i % 10 == 0:
            print i
        cvg.iterate(nsample=500,thin=10,n=50,r=args.r,plot=True,save=True,permutationFunction=permute,burnin=10)
        cvg2.iterate(nsample=500,thin=10,n=50,r=args.r,plot=True,save=True,permutationFunction=permute,burnin=10)

    cvg.save()
    cvg2.save()
