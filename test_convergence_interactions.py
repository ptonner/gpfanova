from test_convergence import ConvergenceTest
import numpy as np
import itertools, gpfanova

class ConvergenceTestInteractions(ConvergenceTest):

    def buildModel(self,k=2,mk=2,r=5,n=50,**kwargs):
        self.x = np.linspace(-1,1,n)[:,None]
        self.y = np.zeros((n,k*mk*r))

        temp = range(mk)
        self.effect = np.array(list(itertools.product(*([temp]*k)))*r)

        #self.m = gpfanova.fanova.FANOVA(self.x,self.y,self.effect,interactions=True,**kwargs)
        self.m = gpfanova.fanova.FANOVA(self.x,self.y,self.effect,**kwargs)
        self.m.y,self.fsamples = self.m.samplePrior()

if __name__ == "__main__":

    import argparse, time
    parser = argparse.ArgumentParser(description='Run convergence test for GP-FANOVA with interactions.')
    # parser.add_argument('strains',metavar=('s'), type=str, nargs='*',
    #                   help='strains to build model for')
    parser.add_argument('-p', dest='partialInteractions', action='store_true',default=False,
                       help='only include partial interactions')

    args = parser.parse_args()

    cvg = ConvergenceTestInteractions('test-interactions-k2-mk2-%s'%time.ctime().replace(" ","-"))
    cvg.addScalarInterval('y_sigma',.05,lambda x: cvg.m.observationLikelihood(sigma=x,prior_lb=-2,prior_ub=2))
    cvg.addScalarInterval('prior0_sigma',.05,lambda x: cvg.m.prior_likelihood(0,sigma=x,prior_lb=-2,prior_ub=2))
    cvg.addScalarInterval('prior0_lengthscale',.05,lambda x: cvg.m.prior_likelihood(0,lengthscale=x,prior_lb=-2,prior_ub=2))
    cvg.addScalarInterval('prior1_sigma',.05,lambda x: cvg.m.prior_likelihood(1,sigma=x,prior_lb=-2,prior_ub=2))
    cvg.addScalarInterval('prior1_lengthscale',.05,lambda x: cvg.m.prior_likelihood(1,lengthscale=x,prior_lb=-2,prior_ub=2))
    cvg.addScalarInterval('prior2_sigma',.05,lambda x: cvg.m.prior_likelihood(2,sigma=x,prior_lb=-2,prior_ub=2))
    cvg.addScalarInterval('prior2_lengthscale',.05,lambda x: cvg.m.prior_likelihood(2,lengthscale=x,prior_lb=-2,prior_ub=2))

    def permute(m):
        #m.parameter_cache['y_sigma'] = np.random.uniform(-1,1)

        m.parameter_cache['prior0_sigma'] = np.random.uniform(-1,1)
        m.parameter_cache['prior0_lengthscale'] = np.random.uniform(-1,1)

        m.parameter_cache['prior1_sigma'] = np.random.uniform(-1,1)
        m.parameter_cache['prior1_lengthscale'] = np.random.uniform(-1,1)

        m.parameter_cache['prior2_sigma'] = np.random.uniform(-1,1)
        m.parameter_cache['prior2_lengthscale'] = np.random.uniform(-1,1)

        return m

    interactions = True
    k = 2

    if args.partialInteractions:
        k = 3
        interactions = [(0,1)]

    cvg.buildModel(k=k,interactions=interactions)
    for f in range(cvg.m.f):
        cvg.addFunctionInterval(f,.95)

    for i in range(50):
        if i % 10 == 0:
            print i
        cvg.iterate(nsample=10000,thin=10,n=10,r=2,y_sigma=-2,k=k,interactions=interactions,plot=True,save=True,permutationFunction=permute,burnin=200)

    cvg.save()
