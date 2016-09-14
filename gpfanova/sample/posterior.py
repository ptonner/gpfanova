import scipy
import numpy as np

class Parameter(object):

    def __init__(self,name,default):
        self.name = name
        self.default = default

class Scalar(Parameter):

    def __init__(self,name,default=0):
        Parameter.__init__(self,name,default)

class Vector(Parameter):

    def __init__(self,name,l=1,default=None):
        self.l=l
        if default is None:
            default = [0]*self.l
        Parameter.__init__(self,name,default)

class ParameterContainer(object):

    def __init__(self,*params):
        self.parameters = params

    def __getitem__(self,key):
        return self.parameters[key]

    def parameterList(self,names=False):
        for k,v in self.parameters.iteritems():
            if names:
                yield v.name
            else:
                yield v

class Sample(ParameterContainer):

    @staticmethod
    def fromOther(n,other):
        if not isinstance(other,Sample):
            raise TypeError('must provide instance of type Sample, %s provided' % str(type(other)))
        s = Sample(n,*other.parameterList())
        for k in other.values.keys():
            s.values[k] = other.values[k]
        return s

    def __init__(self,n,*args):
        self.n = n
        self.values = {}
        self.parameters = {}
        for p in args:
            self.values[p.name] = p.default
            self.parameters[p.name] = p

    def parameter(self,p):
        return self.values[p]

    def __getitem__(self,p):
        return self.parameter(p)

    def __setitem__(self,p,v):
        self.values[p] = v

class Sampler(object):

    def __init__(self,_type,depends=[]):
        """
        Args:
        	depends: depends on curent values of these parameters e.g. slice sampling, MH
        """
        self.type = _type
        self.depends = depends

        self.depends = map(lambda x: x.name if issubclass(type(x),Parameter) else x,self.depends)

    def sample(self,*args):
        # kwargs = {}
        # for i,d in enumerate(self.depends):
        #     kwargs[d] = args[i]

        return self._sample(*args)

    def _sample(self,*args,**kwargs):
        raise NotImplemented("implement this function for your sampler!")

    def __repr__(self):
        if len(self.depends) < 3:
            return "%s: %s" % (self.type, ', '.join(self.depends))
        return "%s: %s, ..." % (self.type, ', '.join(self.depends[:3]))

class Identity(Sampler):

    def __init__(self,param):
        Sampler.__init__(self,'identity',depends=[param])

    def _sample(self,param):
        return param

class Offset(Sampler):

    def __init__(self,param,offset):
        Sampler.__init__(self,'offset',depends=[param])
        self.offset = offset

    def _sample(self,param):
        return param+self.offset

class Function(Sampler):

    @staticmethod
    def functionMatrix(n,*fxns):
        m = np.zeros((n,len(fxns)))
        for i in range(len(fxns)):
            m[:,i] = fxns[i]
        return m

    def __init__(self,name,designMatrix,x,fxns,kernel,obsKernel,obs):
        self.name = name
        if not self.name in fxns:
            raise ValueError("this function must be in the provided list of functions!")

        self.designMatrix = designMatrix
        self.x = x
        self.n = self.x.shape[0]

        self.kernel = kernel
        if not issubclass(type(self.kernel),Kernel):
            raise TypeError('must provide an object of type kernel')

        self.obsKernel = obsKernel
        if not issubclass(type(self.obsKernel),Kernel):
            raise TypeError('must provide an object of type kernel')

        self.obs = obs

        Sampler.__init__(self,'function',depends=fxns)

        self.index = self.depends.index(self.name)

    def residual(self,fm):
        return self.obs.T - np.dot(self.designMatrix,fm.T)

    def _sample(self,*fxns):

        fm = Function.functionMatrix(self.n,*fxns)

        m = self.residual(fm)

        # in order to isolate the function f from the likelihood of each observation
        # we have to divide by the design matrix coefficient, which leaves a
        # multiplication to balance (which happens twice, e.g. squared).

        # get the design matrix coefficient of each observtion for this function
        # squared because it appears twice in the normal pdf exponential
        n = np.power(self.base.designMatrix[:,self.index],2)
        n = n[n!=0]

        missingValues = np.isnan(m)

        # scale each residual contribution by its squared dm coefficient
        m = np.nansum((m.T*n).T,0)

        # sum for computing the final covariance
        n = np.sum(((~missingValues).T*n).T,0)

        y_inv = self.obsKernel.K_inv(self.x)
        f_inv = self.kernel.K_inv(self.x)

        A = n*y_inv + f_inv
        b = np.dot(y_inv,m)

        # chol_A = np.linalg.cholesky(A)
        chol_A = linalg.jitchol(A)
        chol_A_inv = np.linalg.inv(chol_A)
        A_inv = np.dot(chol_A_inv.T,chol_A_inv)

        mu,cov = np.dot(A_inv,b), A_inv

        return scipy.stats.multivariate_normal.rvs(mu,cov)

class Kernel(object):

    def __init__(self):
        pass

    def K(self,X):
        raise NotImplemented("Implement this function for your kernel")

class RBF(Kernel):

    @staticmethod
    def dist(X,lengthscales):

        X = X/lengthscales

        Xsq = np.sum(np.square(X),1)
        r2 = -2.*np.dot(X, X.T) + Xsq[:,None] + Xsq[None,:]
        r2 = np.clip(r2, 0, np.inf)
        return np.sqrt(r2)

    def __init__(self,p=1):
        self.p = p
        self.sigma = 0
        self.lengthscale = [0]*self.p

    def K(self,X):
        sigma = np.exp(self.sigma)
        lengthscale = np.array(map(np.exp,self.lengthscale))

        dist = RBF.dist(X,lengthscale)
        return sigma*np.exp(-.5*dist**2)

class Posterior(ParameterContainer):

    def __init__(self):
        self.parameters = {}
        self.subscriptions = {}
        self.samplers = {}
        self.samples = []

    def addParameter(self,p,sampler):
        if not issubclass(type(p),Parameter):
            raise TypeError('paramter must be of sub-class Parameter, %s provided' % str(type(p)))
        if not issubclass(type(sampler),Sampler):
            raise TypeError('sampler must be of sub-class Sampler, %s provided' % str(type(sampler)))

        if p.name in self.parameters:
            raise AttributeError("Parameter with name %s already given!" % p.name)

        self.parameters[p.name] = p
        self.samplers[p.name] = sampler

    def addSubscriber(self,p,sub,nameMap=None):
        if issubclass(type(p),Parameter):
            p = p.name

        if nameMap is None:
            nameMap = p

        if p in self.subscriptions:
            self.subscriptions[p].append((sub,nameMap))
        else:
            self.subscriptions[p] = [(sub,nameMap)]

    def buildSample(self,n=0):
            return Sample(n,*self.parameterList())

    def sampleParam(self,p,sample=None):
        if sample is None:
            sample = self.samples[-1]

        sampler = self.samplers[p]
        args = []
        for d in sampler.depends:
            args.append(sample[d])
        return sampler.sample(*args)

    def sample(self,):

        if len(self.samples) == 0:
            sample = self.buildSample(0)
        else:
            sample = Sample.fromOther(self.samples[-1].n+1,self.samples[-1])

        params = self.samplers.keys()

        for p in params:
            # sampler = self.samplers[p]
            # args = []
            # for d in sampler.depends:
            #     args.append(sample[d])
            # sample[p] = sampler.sample(*args)
            sample[p] = self.sampleParam(p,sample)
            self.updateSubscriptions(p,sample)

        self.samples.append(sample)

    def updateSubscriptions(self,p=None,sample=None):
        if sample is None:
            sample = self.samples[-1]

        if p is None:
            for p in self.subscriptions.keys():
                for sub,nm in self.subscriptions[p]:
                    # sub.__setattr__(p,sample[p])
                    sub.__dict__[nm] = sample[p]
        else:
            if p in self.subscriptions:
                for sub,nm in self.subscriptions[p]:
                    # sub.__setattr__(p,sample[p])
                    sub.__dict__[nm] = sample[p]
