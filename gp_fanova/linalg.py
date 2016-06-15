import numpy as np
from scipy import linalg
from scipy.linalg import lapack, blas

def jitchol(A, maxtries=5):
	A = np.ascontiguousarray(A)
	L, info = lapack.dpotrf(A, lower=1)
	if info == 0:
		return L
	else:
		diagA = np.diag(A)
		if np.any(diagA <= 0.):
			raise linalg.LinAlgError("not pd: non-positive diagonal elements")
		jitter = diagA.mean() * 1e-6
		num_tries = 1
		while num_tries <= maxtries and np.isfinite(jitter):
			try:
				L = linalg.cholesky(A + np.eye(A.shape[0]) * jitter, lower=True)
				return L
			except:
				jitter *= 10
			finally:
				num_tries += 1
		raise linalg.LinAlgError("not positive definite, even with jitter.")
	import traceback
	try: raise
	except:
		logging.warning('\n'.join(['Added jitter of {:.10e}'.format(jitter),
			'  in '+traceback.format_list(traceback.extract_stack(limit=3)[-2:-1])[0][2:]]))
	return L
