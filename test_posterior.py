import unittest
from gpfanova.sample.posterior import Parameter, ParameterContainer,Scalar,Vector,Sample, Posterior, Sampler, Identity, Offset, Function, RBF
import numpy as np

class PosteriorTests(unittest.TestCase):

    def setUp(self):
        self.post = Posterior()
        self.p = Parameter('test',0)
        self.s = Identity('test')
        self.post.addParameter(self.p,self.s)

    def test_sampling(self):

        self.post.sample()
        self.assertIsInstance(self.post.samples[-1],Sample)

    def test_samplingAdvanced(self):

        samplerIdent = Identity('test')
        samplerOffset = Offset('test',10)

        paramIdent = Parameter('testIdent',-20)
        paramOffset = Parameter('testOffset',-20)

        self.post.addParameter(paramIdent,samplerIdent)
        self.post.addParameter(paramOffset,samplerOffset)

        class Subscriber():
            def __init__(self):
                self.test=None
                self.testIdent=None
                self.testOffset = None
        sub = Subscriber()
        self.post.addSubscriber('test',sub)
        self.post.addSubscriber('testIdent',sub)
        self.post.addSubscriber('testOffset',sub)

        self.post.sample()
        sample = self.post.samples[-1]

        self.assertEquals(sample['testIdent'],0)
        self.assertEquals(sample['testOffset'],10)
        self.assertEquals(sub.test,0)
        self.assertEquals(sub.testIdent,0)
        self.assertEquals(sub.testOffset,10)

    def test_functions(self):
        f = 3
        x = np.linspace(-1,1)
        obs = np.random.normal(size=(50,f))
        designMatrix = np.ones((50,f))

        functionParams = ParameterContainer(*[Vector("f%d"%i,50) for i in range(f)])

        kern = RBF(1)
        obsKern = RBF(1)

        for i in range(f):
            fxn = Function(functionParams[i].name,designMatrix,x,functionParams.parameterList(names=True),kern,obsKern,obs)
            self.post.addParameter(functionParams[i],fxn)

        self.post.addParameter(Scalar('sigma',),Offset('sigma',1))
        self.post.addParameter(Vector('lengthscale',),Identity('lengthscale'))
        self.post.addSubscriber('sigma',kern)
        self.post.addSubscriber('lengthscale',kern)
        self.post.addSubscriber('sigma',obsKern)
        self.post.addSubscriber('lengthscale',obsKern)

        sample = self.post.buildSample()

        for fxn in functionParams:
            self.assertIn(fxn.name,sample.values)
            self.assertIn(fxn.name,sample.parameters)
            self.assertEquals(fxn.default,sample.values[fxn.name])

        self.assertEquals(kern.sigma,sample.values['sigma'])
        self.assertEquals(kern.lengthscale,sample.values['lengthscale'])

        self.post.sample()
        sample = self.post.samples[-1]


    def test_addParameter(self):

        self.assertIn(self.p.name,self.post.parameters)
        self.assertEquals(self.post.parameters[self.p.name],self.p)
        self.assertEquals(self.post.samplers[self.p.name],self.s)

        self.assertRaises(AttributeError,lambda: self.post.addParameter(self.p,self.s))
        self.assertRaises(TypeError,lambda: self.post.addParameter('fda',self.s))
        self.assertRaises(TypeError,lambda: self.post.addParameter(self.p,'dfas'))
        self.assertRaises(TypeError,lambda: self.post.addParameter('fda','fdsa'))

    def test_buildSample(self):
        sample = self.post.buildSample(0)
        self.assertIn(self.p.name,sample.values)
        self.assertIn(self.p.name,sample.parameters)
        self.assertEquals(self.p.default,sample.values[self.p.name])

def main():
    unittest.main()

if __name__ == '__main__':
    main()
