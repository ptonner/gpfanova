import unittest
import numpy as np
import gpfanova
from gpfanova import Prior

def buildData(f):
	x = np.linspace(-1,1)[:,None]
	y = np.zeros((50,f))
	return x,y

class BaseTests(unittest.TestCase):

	def testConstructorWithoutPriors(self):
		f = 1
		x,y = buildData(f)
		m = gpfanova.base.Base(x,y,np.ones(f)[:,None].T)
		self.assertIsNotNone(m)
		self.assertEqual(m.k,1)

		for s in m.buildPriors([([i],'%d'%i) for i in range(10)]):
			self.assertTrue(isinstance(s,Prior))

	def testConstructorWithPriors(self):
		f = 3
		x,y = buildData(f)
		dm = np.eye((f))
		m = gpfanova.base.Base(x,y,dm,priors=[([i],'p%d') for i in range(f)])
		self.assertIsNotNone(m)
		self.assertEqual(m.k,f)

def main():
    unittest.main()

if __name__ == '__main__':
    main()
