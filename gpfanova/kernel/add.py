from kernel import Kernel

class Addition(Kernel):

    def __init__(self,k1,k2,model,*args,**kwargs):
        self.k1 = k1
        self.k2 = k2
        params = []
        for k in [self.k1,self.k2]:
            params.extend(k.parameters)
        Kernel.__init__(self,model,params,*args,**kwargs)

    def _K(self,X,*args,**kwargs):

        kw = {}
        for k in kwargs.keys():
            if k[:3] == 'k1_':
                kw[k[3:]] = kwargs[k]

        k1 = self.k1.K(X,**kw)

        kw = {}
        for k in kwargs.keys():
            if k[:3] == 'k2_':
                kw[k[3:]] = kwargs[k]

        k2 = self.k2.K(X,**kw)

        return k1 + k2
