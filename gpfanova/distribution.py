import parameter, linalg
from parameter import Parameter, Scalar, Vector
import scipy
import numpy as np
from collections import OrderedDict

class Parameterized(object):

    def __init__(self,*args,**kwargs):
        self.parameters = OrderedDict()
        self.addParameters(*args,**kwargs)

    def __getattribute__(self, name):
        parameters = object.__getattribute__(self,'parameters')
        if name in parameters:
            return parameters[name].value
        else:
            return object.__getattribute__(self,name)

    def addParameter(self,p,name=None):
        if issubclass(type(p),Parameter):
            if name is None:
                self.parameters[p.name] = p
            else:
                self.parameters[name] = p
        else:
            if type(p) == float:
                p = Scalar(p,name)
                self.addParameter(p,name)
            elif type(p) == np.ndarray:
                p = Vector(p,name)
                self.addParameter(p,name)
            else:
                raise TypeError("parameter provided is of type %s, not Parameter"%type(p))

    def buildParameters(self,*args,**kwargs):
        ret = [a if not issubclass(type(a),Parameter) else a.value for a in args]

        keys = self.parameters.keys()
        for k in keys[len(ret):]:
            if k in kwargs:
                p = kwargs[k]
                if issubclass(type(p),Parameter):
                    ret.append(p.value)
                else:
                    ret.append(p)
            else:
                ret.append(self.parameters[k].value)

        return ret

    def addParameters(self,*args,**kwargs):
        for a in args:
            self.addParameter(a)
        for k,v in kwargs.iteritems():
            self.addParameter(v,name=k)

class Distribution(object):
    """

    observations<Parameter> <- distribution <- parameters<Parameter>
    """

    def __init__(self,name=None,parameters=[],observations=[],):
        self.name = name
        self.observations = [a for a in observations if issubclass(type(a),Parameter)]

        self.parameters = Parameterized(*parameters)
        # self.parameters = OrderedDict()
        # for a in parameters:
        #     if issubclass(type(a),Parameter):
        #         self.parameters[a.name] = a
                # {(a.name,a) for a in parameters if issubclass(type(a),Parameter)}

    def addParameter(self,p,name=None):
        self.parameters.addParameter(p,name)
        # if issubclass(type(p),Parameter):
        #     if name is None:
        #         self.parameters[p.name] = p
        #     else:
        #         self.parameters[name] = p

    def addObservation(self,o):
        if issubclass(type(o),Parameter):
            self.observation.append(o)

    def loglikelihood(self,variables=None,*args,**kwargs):
        if variables is None:
            variables = []
            for p in self.observations:
                variables.append(p.value)

        # args = list(args)
        # for i in self.parameters.keys()[len(args):]:
        #     p = self.parameters[i]
        #     if p.logspace:
        #         args.append(pow(10,self.parameters[i].value))
        #     else:
        #         args.append(self.parameters[i].value)
        #
        # keys = kwargs.keys()
        # for k in keys:
        #     if k in self.parameters:
        #         i = self.parameters.keys().index(k)
        #         args[i] = kwargs.pop(k)

        args = self.parameters.buildParameters(*args,**kwargs)

        return self._loglikelihood(variables,*args)

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
                ', '.join([str(d) for d in self.parameters.parameters.items()]))

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

class Kernel(Parameterized):

    def __init__(self,p,*args,**kwargs):
        self.p = p
        # self.parameters = [a for a in args if issubclass(type(a),Parameter)]
        Parameterized.__init__(self,*args,**kwargs)

    def K(self,x,*args,**kwargs):
        params = self.buildParameters(*args,**kwargs)
        return self._K(x,*params)

    def _K(self,x,*args,**kwargs):
        raise NotImplementedError("implement this for your kernelv!")

    def K_inv(self,X,*args,**kwargs):

        K = self.K(X,*args,**kwargs)
        # K += np.eye(K.shape[0])*K.diagonal().mean()*1e-6

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
            params.extend([v for k,v in k.parameters.items()])

        assert self.k1.p == self.k2.p

        Kernel.__init__(self,self.k1.p,*params,**kwargs)

    def _K(self,X,*args):

        k1 = self.k1.K(X,*args[:len(self.k1.parameters)])
        k2 = self.k2.K(X,*args[len(self.k1.parameters):])

        return k1 + k2

class Product(Kernel):

    def __init__(self,k1,k2,*args,**kwargs):
        self.parameters = {}

        self.k1 = k1
        self.k2 = k2
        params = []
        for k in [self.k1,self.k2]:
            params.extend([v for k,v in k.parameters.items()])

        assert self.k1.p == self.k2.p

        Kernel.__init__(self,self.k1.p,*params,**kwargs)

    def _K(self,X,*args):

        k1 = self.k1.K(X,*args[:len(self.k1.parameters)])
        k2 = self.k2.K(X,*args[len(self.k1.parameters):])

        return k1 * k2

class RBF(Kernel):

    @staticmethod
    def dist(X,lengthscale):
        X = X/lengthscale

        Xsq = np.sum(np.square(X),1)
        r2 = -2.*np.dot(X, X.T) + Xsq[:,None] + Xsq[None,:]
        r2 = np.clip(r2, 0, np.inf)
        return np.sqrt(r2)

    def _K(self,X,sigma=None,lengthscale=None):
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

class RBF_LinearVariance(Kernel):

    def _K(self,X,sigma1,sigma2,lengthscale):

        dist = RBF.dist(X,lengthscale)
        # return (sigma1 + sigma2*X[:,0])*np.exp(-.5*dist**2)
        out = np.dot(X,X.T)
        return (sigma1 + sigma2*out)*np.exp(-.5*dist**2)

    def __init__(self,p,sigma1=None,sigma2=None,lengthscale=None,name='rbf-linearVariance'):
        # if sigma1 is None:
        #     sigma1 = Scalar(1,'%s-sigma1'%name)
		# if sigma2 is None:
        #     sigma2 = Scalar(1,'%s-sigma2'%name)
        # elif not issubclass(type(sigma),Parameter):
        #     sigma = Scalar(sigma,'%s-sigma'%name)

        # if lengthscale is None:
        #     lengthscale = Vector(np.ones(p),'%s-lengthscale'%name)
        # elif not issubclass(type(lengthscale),Parameter):
        #     lengthscale = Vector(lengthscale,'%s-lengthscale'%name)

        # self.sigma = sigma
        # self.lengthscale = lengthscale

        Kernel.__init__(self,p,)
        self.addParameter(sigma1,'sigma1')
        self.addParameter(sigma2,'sigma2')
        self.addParameter(lengthscale,'lengthscale')

class Linear(Kernel):

    @staticmethod
    def dist(X,lengthscale):
        X = X/lengthscale

        Xsq = np.sum(np.square(X),1)
        r2 = -2.*np.dot(X, X.T) + Xsq[:,None] + Xsq[None,:]
        r2 = np.clip(r2, 0, np.inf)
        return np.sqrt(r2)

    def _K(self,X,sigma,lengthscale):

        X = X/lengthscale
        # return (sigma1 + sigma2*X[:,0])*np.exp(-.5*dist**2)
        out = np.dot(X,X.T)
        return sigma*out

    def __init__(self,p,sigma=None,lengthscale=None,name='linear'):

        Kernel.__init__(self,p,)
        self.addParameter(sigma,'sigma')
        self.addParameter(lengthscale,'lengthscale')


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

        parameters.extend([v for k,v in self.kernel.parameters.items()])

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
        for f in self.parameters.parameters.keys()[:-len(self.kernel.parameters)]:
            a.append(self.parameters.parameters[f].value)
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
