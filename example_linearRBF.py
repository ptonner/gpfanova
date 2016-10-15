from gpfanova.distribution import GaussianProcess, HierarchicalGP, Normal, RBF_LinearVariance, White
from gpfanova.parameter import Vector, Scalar
from gpfanova.sampler import FunctionSampler, MCMC
import scipy
import numpy as np, pandas as pd

# parameters

x = np.linspace(0.1,1)[:,None]

r = .01
r2 = .1
t = 1
kern = RBF_LinearVariance(1,r*t,(1-r)*t,np.array([.5]))
kern2 = White(1,r2*t)

cov = kern.K(x)

f = Vector(scipy.stats.multivariate_normal.rvs(np.zeros(50),cov),'f')
# f2 = Vector(scipy.stats.multivariate_normal.rvs(np.zeros(50),cov),'f2')

fsave = f.value
# f2save = f2.value

y = Vector(scipy.stats.multivariate_normal.rvs(f.value,kern2.K(x)),'y')
# y2 = Vector(scipy.stats.multivariate_normal.rvs(f.value+f2.value,kern2.K(x)),'y2')

# priors/likelihood

gp_f = GaussianProcess(kern,x,observations=[f])

# gp_f2 = GaussianProcess(kern,x,observations=[f2])
gp_y = HierarchicalGP(kern2,x,[f],[1],observations=[y])
# gp_y2 = HierarchicalGP(kern2,x,[f,f2],[1,1],observations=[y2])

sigma1f_norm = Normal(0,1,name='sigma1f_norm',logspace=True,observations=[kern.sigma1])
sigma2f_norm = Normal(0,1,name='sigma2f_norm',logspace=True,observations=[kern.sigma2])
lf_norm = Normal(0,1,name='lf_norm',logspace=True,observations=[kern.lengthscale])
sigmay_norm = Normal(0,1,name='sigmay_norm',logspace=True,observations=[kern2.sigma])

# samplers

fsample = FunctionSampler(f,gp_f,[gp_y],x)

# sigmaf_sample = Slice(kern.sigma,sigmaf_norm,[gp_f])
# lf_sample = Slice(kern.lengthscale,sigmaf_norm,[gp_f])
# sigmay_sample = Slice(kern2.sigma,sigmay_norm,[gp_y])

mcmc = MCMC(fsample)

for i in range(500):
    mcmc.sample()

samples = pd.DataFrame(mcmc.iterations)

fsamples = np.array([list(z) for z in list(samples.f.values)])
