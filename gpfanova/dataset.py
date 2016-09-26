import pandas as pd
import numpy as np
import os

class DataSet(object):

    def __init__(self,dir,meta='meta.csv',data='data.csv',timecol=0):
        if not all([x in os.listdir(dir) for x in [meta,data]]):
            raise AttributeError("must provide data with %s and %s!"%(meta,data))

        self.meta = pd.read_csv(os.path.join(dir,meta))
        self.data = pd.read_csv(os.path.join(dir,data),index_col=timecol)

        assert self.data.shape[1] == self.meta.shape[0], 'frames do no match'

        self.data.columns = self.meta.index

    def build(self,effects=[],covariates=[]):

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

            x = pd.DataFrame(np.array(pivot.index.tolist()),columns=['x']+covariates)
            y = pivot.values

        else:
            x = pd.DataFrame(self.data.index.values,columns=['x'])
            y = self.data.values

        effect = self.meta[effects]

        return x,y,effect
