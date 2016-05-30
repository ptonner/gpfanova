from kernel import Kernel
import numpy as np

class White(Kernel):
	"""White noise kernel"""

	def _K(self,X,sigma):

		return sigma*np.eye(X.shape[0])
