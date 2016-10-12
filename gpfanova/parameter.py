import numpy as np

class Parameter(object):

    def __init__(self,value=None,name=None,type=None,fixed=False,distribution=None,logspace=False,*args,**kwargs):
        self.type = type

        self.value = value
        self.name = name

        self.fixed = fixed
        self.distribution = distribution
        self.logspace = logspace
        self.depends = [a for a in args if issubclass(type(a),Parameter)]

    def __repr__(self,):
        return '%s (%s): %s'%(self.name,self.type,self.value)

    def __setattr__(self, name, value):
        if name == 'value':
            if not type(value) == self.type:
                # self.__dict__[name] = self.type(value)
                value = self.type(value)

        self.__dict__[name] = value

            # assert type(value) == self.type, 'must provide value of type %s!'%self.type

    def __eq__(self,other):
        return type(self) == type(other) and self.value == other.value

class Scalar(Parameter):

    def __init__(self,value,name='a',*args,**kwargs):
        Parameter.__init__(self,value,name,type=float,*args,**kwargs)

class Vector(Parameter):
    def __init__(self,value,name='a',*args,**kwargs):
        Parameter.__init__(self,value,name,type=np.array,*args,**kwargs)
