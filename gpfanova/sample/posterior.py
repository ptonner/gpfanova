from sampler import Sampler

class Parameter(object):

    def __init__(self,name,default):
        self.name = name
        self.default = default

class Scalar(Parameter):

    def __init__(self,name,default=0):
        Parameter.__init__(self,name,default)

class Vector(Parameter):

    def __init__(self,name,l,default=None):
        self.l=l
        if default is None:
            default = [0]*self.l
        Parameter.__init__(self,name,default)

class ParameterContainer(object):

    def parameterList(self):
        for k,v in self.parameters.iteritems():
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

    def addSubscriber(self,p,sub):
        if p in self.subscriptions:
            self.subscriptions[p.name].append(sub)
        else:
            self.subscriptions[p.name] = [sub]

    def buildSample(self,n=0,f=None):
        if not f is None:
            return Sample(f=f)
        else:
            return Sample(n,*self.parameterList())

    def updateSubscriptions(self,p=None):
        if p is None:
            for p in self.subscriptions.keys():
                for sub in self.subscriptions[p]:
                    sub.__setattr__(p,self.currentValue(p))
        else:
            for sub in self.subscriptions[p]:
                sub.__setattr__(p,self.currentValue(p))
