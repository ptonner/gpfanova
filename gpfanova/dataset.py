import pandas as pd
import numpy as np
import os

class DataSet(object):

    def __init__(self,dir,meta='meta.csv',data='data.csv',timecol=0):
        if not all([x in os.listdir(dir) for x in [meta,data]]):
            raise AttributeError("must provide data with %s and %s!"%(meta,data))

        self.meta = pd.read_csv(os.path.join(dir,meta))
        self.data = pd.read_csv(os.path.join(dir,data),index_col=timecol)

        assert self.data.shape[1] == self.meta.shape[0], 'frames do no match, %d x %d' % (self.data.shape[1], self.meta.shape[0])

        self.data.columns = self.meta.index

    def build(self,effects=[],covariates=[],scale=None,**kwargs):

        if 'x' in covariates:
            covariates.remove('x')

        if len(covariates)>0:
            temp = pd.concat((self.meta,self.data.T),1)
            temp['rep'] = temp.index

            tidy = pd.melt(temp,self.meta.columns.tolist()+['rep'],
                            self.data.index.tolist(),
                            var_name='x',value_name='y')

            pivot = pd.pivot_table(tidy,values='y',
                            index=['x']+covariates,
                            columns=['rep'])

            x = pd.DataFrame(np.array(pivot.index.tolist()),columns=['x']+covariates).values
            y = pivot.values

        else:
            x = pd.DataFrame(self.data.index.values,columns=['x']).values
            y = self.data.values

        effect = self.meta[effects]
        labels = []

        select = [True]*self.meta.shape[0]
        for k in kwargs.keys():
            if k in self.meta:
                if type(kwargs[k]) == list:
                    select = (select) & (self.meta[k].apply(lambda x: x in kwargs[k]))
                else:
                    select = (select) & (self.meta[k] == kwargs[k])
        y = y[:,np.where(select)[0]]
        effect = effect.loc[select,:]

        for e in effect.columns:
            temp,l = pd.factorize(effect[e])
            effect[e] = temp
            labels.append(l.tolist())

        if scale=='range':
            x = (x-x.min())/(x.max()-x.min())
        elif scale=='norm':
            x = (x-x.mean())/x.std()

        return x,y,effect,labels
