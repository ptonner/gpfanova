from kernel import Kernel

class Addition(Kernel):

    def __init__(self,k1,k2,model,*args,**kwargs):
        self.k1 = k1
        self.k2 = k2
        params = []
        for k in [self.k1,self.k2]:
            params.extend(k.parameters)
        Kernel.__init__(self,model,params,*args,**kwargs)

    def _K(self,X,*args):

        k1 = self.k1._K(X,*args[:len(self.k1.parameters)])
        k2 = self.k2._K(X,*args[len(self.k1.parameters):])

        return k1 + k2
