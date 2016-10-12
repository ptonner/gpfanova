from gpfanova.distribution import GaussianProcess, HierarchicalGP, Normal, RBF, White
from gpfanova.parameter import Vector, Scalar
from gpfanova.sampler import FunctionSampler, MCMC
import scipy
import numpy as np, pandas as pd

# parameters

x = np.linspace(-5,5)[:,None]
kern = RBF(1,.5,2)
kern2 = White(1,.1)
cov = kern.K(x)

f = Vector(scipy.stats.multivariate_normal.rvs(np.zeros(50),cov),'f')
f2 = Vector(scipy.stats.multivariate_normal.rvs(np.zeros(50),cov),'f2')

fsave = f.value
f2save = f2.value

y = Vector(scipy.stats.multivariate_normal.rvs(f.value,kern2.K(x)),'y')
y2 = Vector(scipy.stats.multivariate_normal.rvs(f.value+f2.value,kern2.K(x)),'y2')

# priors/likelihood

gp_f = GaussianProcess(kern,x,observations=[f])
gp_f2 = GaussianProcess(kern,x,observations=[f2])
gp_y = HierarchicalGP(kern2,x,[f],[1],observations=[y])
gp_y2 = HierarchicalGP(kern2,x,[f,f2],[1,1],observations=[y2])

sigmaf_norm = Normal(0,1,name='sigmaf_norm',logspace=True,observations=[kern.sigma])
lf_norm = Normal(0,1,name='lf_norm',logspace=True,observations=[kern.lengthscale])
sigmay_norm = Normal(0,1,name='sigmay_norm',logspace=True,observations=[kern2.sigma])

# samplers

fsample = FunctionSampler(f,gp_f,[gp_y,gp_y2],x)
f2sample = FunctionSampler(f2,gp_f2,[gp_y2],x)

# sigmaf_sample = Slice(kern.sigma,sigmaf_norm,[gp_f])
# lf_sample = Slice(kern.lengthscale,sigmaf_norm,[gp_f])
# sigmay_sample = Slice(kern2.sigma,sigmay_norm,[gp_y])

mcmc = MCMC(fsample,f2sample)

for i in range(5000):
    mcmc.sample()

samples = pd.DataFrame(mcmc.iterations)

fsamples = np.array([list(z) for z in list(samples.f.values)])
f2samples = np.array([list(z) for z in list(samples.f2.values)])
