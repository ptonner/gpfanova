import parameter, linalg
from parameter import Parameter, Scalar, Vector
import scipy
import numpy as np
from collections import OrderedDict

class Distribution(object):
    """

    observations<Parameter> <- distribution <- parameters<Parameter>
    """

    def __init__(self,name=None,parameters=[],observations=[],):
        self.name = name
        self.observations = [a for a in observations if issubclass(type(a),Parameter)]
        self.parameters = OrderedDict()
        for a in parameters:
            if issubclass(type(a),Parameter):
                self.parameters[a.name] = a
                # {(a.name,a) for a in parameters if issubclass(type(a),Parameter)}

    def addParameter(self,p,name=None):
        if issubclass(type(p),Parameter):
            if name is None:
                self.parameters[p.name] = p
            else:
                self.parameters[name] = p

            # self.parameters.append(p)

    def addObservation(self,o):
        if issubclass(type(o),Parameter):
            self.observation.append(o)

    def loglikelihood(self,variables=None,*args,**kwargs):
        if variables is None:
            variables = []
            for p in self.observations:
                variables.append(p.value)

        args = list(args)
        for i in self.parameters.keys()[len(args):]:
            p = self.parameters[i]
            if p.logspace:
                args.append(pow(10,self.parameters[i].value))
            else:
                args.append(self.parameters[i].value)

        keys = kwargs.keys()
        # for k,v in kwargs.iteritems():
        for k in keys:
            if k in self.parameters:
                # p = self.parameters[k]
                i = self.parameters.keys().index(k)
                args[i] = kwargs.pop(k)

                # if p.logspace:
                #     args[i] = pow(args[i],10)

        return self._loglikelihood(variables,*args,**kwargs)

    def _loglikelihood(self,y,*args,**kwargs):
        raise NotImplementedError("implement this for your distribution!")

    def sample(self,observationConditional=True,observations=[],*args,**kwargs):

        if observationConditional:
            observations.extend([p.value for p in self.observations])

        observations += args

        return self._sample(*observations,**kwargs)

    def _sample(self,*args,**kwargs):
        raise NotImplementedError("implement this for your distribution!")

    def __repr__(self,):
        return '''%s (%s)
        observations: %s
        parameters: %s'''% \
                (self.name,type(self),
                ', '.join([str(p) for p in self.observations]),
                ', '.join([str(d) for d in self.parameters.items()]))

class Normal(Distribution):

    def __init__(self,mu=0,sigma=1,name='normal',logspace=False,*args,**kwargs):

        if not issubclass(type(mu),Parameter):
            mu = Scalar(mu,'%s-mu'%name,logspace=logspace)
        if not issubclass(type(sigma),Parameter):
            sigma = Scalar(sigma,'%s-sigma'%name,logspace=logspace)

        self.mu = mu
        self.sigma = sigma

        #Distribution.__init__(self,name,*args,parameters=[mu,sigma],**kwargs)
        Distribution.__init__(self,*args,**kwargs)
        self.addParameter(self.mu,'mu')
        self.addParameter(self.sigma,'sigma')

    def _loglikelihood(self,y,mu=None,sigma=None,*args,**kwargs):

        y = np.array(y)
        y = y.ravel()

        rv = scipy.stats.norm(mu,sigma)
        return rv.logpdf(y).sum()

    def _sample(self,y,mu=None,sigma=None,*args,**kwargs):
        y = np.array(y)
        y = y.ravel()

        n = y.shape[0]
        rv = scipy.stats.norm(n*y.mean())

class Kernel(object):

    def __init__(self,p,*args,**kwargs):
        self.p = p
        self.parameters = [a for a in args if issubclass(type(a),Parameter)]

    def K(self,x,*args,**kwargs):
        raise NotImplementedError("implement this for your kernelv!")

    def K_inv(self,X,*args,**kwargs):

        K = self.K(X,*args,**kwargs)# + np.eye(X.shape[0])*OFFSET
        try:
            chol = linalg.jitchol(K)
            chol_inv = np.linalg.inv(chol)
        except np.linalg.linalg.LinAlgError,e:
            logger = logging.getLogger(__name__)
            logger.error('Kernel inversion error: %s'%str(self.parameters))
            raise(e)
        inv = np.dot(chol_inv.T,chol_inv)

        return inv

class Addition(Kernel):

    def __init__(self,k1,k2,*args,**kwargs):
        self.k1 = k1
        self.k2 = k2
        params = []
        for k in [self.k1,self.k2]:
            params.extend(k.parameters)

        assert self.k1.p == self.k2.p

        Kernel.__init__(self,self.k1.p,*params,**kwargs)

    def K(self,X,*args):

        k1 = self.k1.K(X,*args[:len(self.k1.parameters)])
        k2 = self.k2.K(X,*args[len(self.k1.parameters):])

        return k1 + k2


class RBF(Kernel):

    @staticmethod
    def dist(X,lengthscale):
        X = X/lengthscale

        Xsq = np.sum(np.square(X),1)
        r2 = -2.*np.dot(X, X.T) + Xsq[:,None] + Xsq[None,:]
        r2 = np.clip(r2, 0, np.inf)
        return np.sqrt(r2)

    def K(self,X,sigma=None,lengthscale=None):
        if sigma is None:
            sigma = self.sigma.value
        if lengthscale is None:
            lengthscale = self.lengthscale.value

        dist = RBF.dist(X,lengthscale)
        return sigma*np.exp(-.5*dist**2)

    def __init__(self,p,sigma=None,lengthscale=None,name='rbf'):
        if sigma is None:
            sigma = Scalar(1,'%s-sigma'%name)
        elif not issubclass(type(sigma),Parameter):
            sigma = Scalar(sigma,'%s-sigma'%name)

        if lengthscale is None:
            lengthscale = Vector(np.ones(p),'%s-lengthscale'%name)
        elif not issubclass(type(lengthscale),Parameter):
            lengthscale = Vector(lengthscale,'%s-lengthscale'%name)

        self.sigma = sigma
        self.lengthscale = lengthscale

        Kernel.__init__(self,p,sigma,lengthscale)

class White(Kernel):
    """White noise kernel"""

    def __init__(self,p,sigma=None,name='white'):
        if sigma is None:
            sigma = Scalar(1,'%s-sigma'%name)
        elif not issubclass(type(sigma),Parameter):
            sigma = Scalar(sigma,'%s-sigma'%name)

        self.sigma = sigma

        Kernel.__init__(self,p,sigma)


    def K(self,X,sigma=None):
        if sigma is None:
            sigma = self.sigma.value

        return sigma*np.eye(X.shape[0])


class GaussianProcess(Distribution):

    def __init__(self,kernel,x,mu=None,parameters=[],name='GP',*args,**kwargs):
        self.kernel = kernel
        self.x = x

        if mu is None:
            mu = np.zeros(self.x.shape[0])
        self.mu = mu

        parameters.extend(self.kernel.parameters)

        Distribution.__init__(self,*args,parameters=parameters,name=name,**kwargs)

    def _loglikelihood(self,y,*args,**kwargs):
        cov = self.kernel.K(self.x)
        return scipy.stats.multivariate_normal.logpdf(y,self.mu,cov,allow_singular=True).sum()

class HierarchicalGP(GaussianProcess):

    def __init__(self,kernel,x,parents,beta,*args,**kwargs):
        GaussianProcess.__init__(self,kernel,x,parameters=parents,name='GP-Hierarchical',*args,**kwargs)
        self.beta = np.array(beta)

    def functionMatrix(self):
        a = []
        for f in self.parameters.keys()[:-len(self.kernel.parameters)]:
            a.append(self.parameters[f].value)
        return np.column_stack(tuple(a))

    def residual(self,remove=[]):
        a = self.functionMatrix()
        mu = np.dot(a,self.beta)

        if len(remove)>0:
            mu -= np.dot(a[:,remove],[self.beta[i] for i in range(len(self.beta)) if i in remove])

        return np.column_stack(tuple([y.value-mu for y in self.observations]))

    def _loglikelihood(self,y,*args,**kwargs):

        mu = np.dot(np.column_stack(args[:-len(self.kernel.parameters)]),self.beta)
        cov = self.kernel.K(self.x)
        return scipy.stats.multivariate_normal.logpdf(y,mu,cov,allow_singular=True).sum()
