import gp_fanova
import model
import numpy as np
import scipy
import matplotlib.pyplot as plt

x = np.linspace(-1,1)[:,None]
y = np.zeros((50,6))
effect = np.array([0]*3+[1]*3)
# effect = np.array([0]*3+[1]*3+[2]*3+[3]*3+[4]*3)

cov = np.eye(50)*.1

y[:,:3] = gp_fanova._mu_sample + scipy.stats.multivariate_normal.rvs(gp_fanova.alpha_samples[:,0],cov,3).T
y[:,3:6] = gp_fanova._mu_sample + scipy.stats.multivariate_normal.rvs(gp_fanova.alpha_samples[:,1],cov,3).T
# y[:,6:9] = gp_fanova._mu_sample + scipy.stats.multivariate_normal.rvs(gp_fanova.alpha_samples[:,2],cov,3).T
# y[:,9:12] = gp_fanova._mu_sample + scipy.stats.multivariate_normal.rvs(gp_fanova.alpha_samples[:,3],cov,3).T
# y[:,12:] = gp_fanova._mu_sample + scipy.stats.multivariate_normal.rvs(gp_fanova.alpha_samples[:,3],cov,3).T

m = model.GP_FANOVA(x,y,effect)
