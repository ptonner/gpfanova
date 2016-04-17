import gp_fanova
import model
import numpy as np
import scipy
import matplotlib.pyplot as plt

x = np.linspace(-1,1)[:,None]
y = np.zeros((50,9))
effect = np.array([0]*3+[1]*3+[2]*3)

cov = np.eye(50)*.1

y[:,:3] = scipy.stats.multivariate_normal.rvs(gp_fanova.alpha_samples[:,0],cov,3).T
y[:,3:6] = scipy.stats.multivariate_normal.rvs(gp_fanova.alpha_samples[:,1],cov,3).T
y[:,6:] = scipy.stats.multivariate_normal.rvs(gp_fanova.alpha_samples[:,2],cov,3).T

m = model.GP_FANOVA(x,y,effect)
