import unittest
from gpfanova.sample.posterior import Parameter,Scalar,Vector,Sample, Posterior, Sampler, Identity, Offset

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
