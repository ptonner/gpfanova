# from sampler import Sampler

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

class Sampler(object):

    def __init__(self,_type,depends=[]):
        """
        Args:
        	depends: depends on curent values of these parameters e.g. slice sampling, MH
        """
        self.type = _type
        self.depends = depends

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
            self.subscriptions[p].append(sub)
        else:
            self.subscriptions[p] = [sub]

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
                for sub in self.subscriptions[p]:
                    # sub.__setattr__(p,sample[p])
                    sub.__dict__[p] = sample[p]
        else:
            if p in self.subscriptions:
                for sub in self.subscriptions[p]:
                    # sub.__setattr__(p,sample[p])
                    sub.__dict__[p] = sample[p]
