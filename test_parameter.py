import unittest
# import numpy as np
# import gpfanova
# from gpfanova import Prior
from gpfanova.sample.posterior import Parameter,Scalar,Vector

class ParameterTests(unittest.TestCase):

	def testBasicParameter(self):
		p = Parameter('test',0)

    def testScalarParameter(self):
        p = Scalar('test',default=1)

        self.assertTrue(p.default == 1)

    def testVectorParameter(self):
        p = Vector('test')

        self.assertIsInstance(p.default,list)

def main():
    unittest.main()

if __name__ == '__main__':
    main()
