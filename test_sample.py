import unittest
# import numpy as np
# import gpfanova
# from gpfanova import Prior
from gpfanova.sample.posterior import Parameter,Scalar,Vector,Sample

class SampleTests(unittest.TestCase):

    def setUp(self):
        self.p = Parameter('test',0)
        self.s = Sample(0,self.p)

    def test_Modification(self):
        self.s[self.p.name] = -1
        self.assertEquals(self.s[self.p.name],-1)
        self.assertEquals(self.s.values[self.p.name],-1)

    def test_Copy(self):
        copy = Sample.fromOther(1,self.s)
        self.assertIn(self.p.name,copy.parameters)
        self.assertIn(self.p.name,copy.values)

        self.s.values[self.p.name] = 1
        copy = Sample.fromOther(1,self.s)
        self.assertEquals(copy.values[self.p.name],self.s.values[self.p.name])

def main():
    unittest.main()

if __name__ == '__main__':
    main()
