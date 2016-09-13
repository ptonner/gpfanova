import unittest
import numpy as np
# import gpfanova
from gpfanova.prior import Prior
from gpfanova.sample import Function, FunctionDerivative, Slice

def buildData(f):
	x = np.linspace(-1,1)[:,None]
	y = np.zeros((50,f))
	return x,y

class PriorTests(unittest.TestCase):

	def buildPrior(self,functions=[0],name='test',base=None,p=1,derivatives=False,f=1,n=10,x=range(10),kernel=None):
		return Prior(functions,name,base,p,derivatives,f,n,x,kernel)

	def testConstructor(self):
		name = 'test'
		base = None
		derivatives=False
		kernel=None

		for f in range(10):
			for nf in range(f):
				functions = range(nf)
				for _p in range(5):
					for n in range(10,100,20):
						x = np.arange(n)
						p = Prior(functions,name,base,_p,derivatives,f,n,x,kernel)
						self.assertIsNotNone(p)
						self.assertIsInstance(p,Prior)

	def testBuildSamplers(self):

		p = self.buildPrior()
		for s in p.buildSamplers():
			self.assertTrue(isinstance(s,Function) or isinstance(s,FunctionDerivative) or isinstance(s,Slice))

	# def testConstructorStringFxnNames(self):
	# 	name = 'test'
	# 	base = None
	# 	derivatives=False
	# 	kernel=None
	#
	# 	for f in range(10):
	# 		functions = ['test%d'%i for i in range(f)]
	# 		for _p in range(5):
	# 			for n in range(10,100,5):
	# 				x = np.arange(n)
	# 				p = gpfanova.prior.Prior(functions,name,base,_p,derivatives,f,n,x,kernel)
	# 				self.assertIsNotNone(p)
	# 				self.assertIsInstance(p,gpfanova.prior.Prior)

def main():
    unittest.main()

if __name__ == '__main__':
    main()
