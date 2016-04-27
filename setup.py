import random
random.seed(123)

import gp_fanova
import model
import numpy as np
import scipy, pyDOE
import matplotlib.pyplot as plt



x = np.linspace(-1,1)[:,None]
y = np.zeros((50,12))
# effect = np.array([0]*3+[1]*3)
effect = np.array([0]*3+[1]*3+[2]*3+[3]*3)[:,None]
# effect = np.column_stack(([0]*2+[1]*2+[0]*2+[1]*2,))
# effect = np.array(pyDOE.fullfact([2,2]).tolist()*3).astype(int)

cov = np.eye(50)*.1

y[:,:3] = gp_fanova._mu_sample + scipy.stats.multivariate_normal.rvs(gp_fanova.alpha_samples[:,0],cov,3).T
y[:,3:6] = gp_fanova._mu_sample + scipy.stats.multivariate_normal.rvs(gp_fanova.alpha_samples[:,1],cov,3).T
y[:,6:9] = gp_fanova._mu_sample + scipy.stats.multivariate_normal.rvs(gp_fanova.alpha_samples[:,2],cov,3).T
y[:,9:12] = gp_fanova._mu_sample + scipy.stats.multivariate_normal.rvs(gp_fanova.alpha_samples[:,3],cov,3).T
# y[:,12:] = gp_fanova._mu_sample + scipy.stats.multivariate_normal.rvs(gp_fanova.alpha_samples[:,3],cov,3).T

# y[:,:3] = scipy.stats.multivariate_normal.rvs(gp_fanova.alpha_samples[:,0],cov,3).T
# y[:,3:6] = scipy.stats.multivariate_normal.rvs(gp_fanova.alpha_samples[:,1],cov,3).T

m = model.GP_FANOVA(x,y,effect)

# def sample_data(nk):
# 	y = np.zeros((50,sum(nk)))
#
# 	for k,nextk in zip(nk[:-1],nk[1:]):
# 		y[:,k:nextk]
