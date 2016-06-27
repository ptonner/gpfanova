# import random
# random.seed(123)

import numpy as np
import pandas as pd
import scipy, pyDOE
import matplotlib.pyplot as plt
import gpfanova



# x = np.linspace(-1,1)[:,None]
# y = np.zeros((50,12))
# # effect = np.array([0]*3+[1]*3)
# effect = np.array([0]*3+[1]*3+[2]*3+[3]*3)[:,None]
# # effect = np.column_stack(([0]*2+[1]*2+[0]*2+[1]*2,))
# effect = np.array(pyDOE.fullfact([2,2]).tolist()*3).astype(int)
# effect = np.array(pyDOE.fullfact([3,2]).tolist()*2).astype(int)
#
# cov = np.eye(50)*.1
#
# y[:,:3] = gpfanova._mu_sample + scipy.stats.multivariate_normal.rvs(gpfanova.alpha_samples[:,0],cov,3).T
# y[:,3:6] = gpfanova._mu_sample + scipy.stats.multivariate_normal.rvs(gpfanova.alpha_samples[:,1],cov,3).T
# y[:,6:9] = gpfanova._mu_sample + scipy.stats.multivariate_normal.rvs(gpfanova.alpha_samples[:,2],cov,3).T
# y[:,9:12] = gpfanova._mu_sample + scipy.stats.multivariate_normal.rvs(gpfanova.alpha_samples[:,3],cov,3).T
# # y[:,12:] = gpfanova._mu_sample + scipy.stats.multivariate_normal.rvs(gpfanova.alpha_samples[:,3],cov,3).T
#
# # y[:,:3] = scipy.stats.multivariate_normal.rvs(gpfanova.alpha_samples[:,0],cov,3).T
# # y[:,3:6] = scipy.stats.multivariate_normal.rvs(gpfanova.alpha_samples[:,1],cov,3).T
#
# m = model.gpfanova(x,y,effect)

# def sample_data(nk):
# 	y = np.zeros((50,sum(nk)))
#
# 	for k,nextk in zip(nk[:-1],nk[1:]):
# 		y[:,k:nextk]


def one_effect_data(effects=3,n=50,r=2,add_fake_effect=False):
	samples = r
	x = np.linspace(-1,1,n)[:,None]
	y = np.zeros((n,effects*samples))
	# effect = np.array([0]*3+[1]*3)
	e = []
	for i in range(effects):
		e += [i]*samples

	effect = np.array(e)[:,None]

	m = gpfanova.fanova.FANOVA(x,y,effect,helmert_covert=True)
	y,_ = m.samplePrior()
	# y,_ = m.samplePrior()

	if add_fake_effect:
		temp = np.zeros((6,2))
		temp[:,0] = effect
		temp[:,1] = np.random.choice(effect,6,replace=False)
		effect = temp

	return x,y,effect

def two_effect_data(e1=2,e2=2,n=3,**kwargs):
	x = np.linspace(-1,1)[:,None]
	# y = np.zeros((50,(e1+e2)*n))
	effect = np.array(pyDOE.fullfact([e1,e2]).tolist()*n).astype(int)
	y = np.zeros((50,effect.shape[0]))

	m = gpfanova.fanova.FANOVA(x,y,effect,**kwargs)
	y,f_samples = m.samplePrior()

	return x,y,effect,f_samples

def multiple_effects(effects=[2,2],m=3,n=50,fullFactorial=True,seed=False,**kwargs):
	x = np.linspace(-1,1,n)[:,None]
	# y = np.zeros((50,(e1+e2)*n))

	if seed:
		np.random.seed(123)

	if fullFactorial:
		effect = np.array(pyDOE.fullfact(effects).tolist()*m).astype(int)
	else:
		effect = np.array([[np.random.choice(range(e)) for e in effects] for i in range(m)])
	y = np.zeros((n,effect.shape[0]))

	m = gpfanova.fanova.FANOVA(x,y,effect,**kwargs)
	y,f_samples = m.samplePrior()

	return x,y,effect,f_samples

def hsalinarum_TF(strains=[],standard=False,paraquat=False,osmotic=False,heatshock=False):
	import os
	datadir = os.path.abspath(os.path.join(os.path.abspath(__file__), os.pardir,os.pardir,'data'))
	# print datadir
	# print os.path.join(datadir,"hsalinarum/tidy_normalize_log_st0.csv")

	data = pd.read_csv(os.path.join(datadir,"hsalinarum/tidy_normalize_log_st0.csv"),index_col=None)
	conditions = ['Experiment','Well','Strain','standard','paraquat','osmotic','heatshock']
	temp = data.set_index(conditions+['time'])
	temp = temp[['OD']]

	if len(strains)==0:
		strains = ['ura3', 'hlx1', 'asnC', 'trh2', 'trh3', 'trh4', 'copR', 'kaiC',
       'idr1', 'idr2', 'troR', 'phoU', 'prp2', 'birA', 'trmB', 'arcR',
       'VNG0039', 'VNG2268', 'VNG0471', 'VNG1029', 'VNG2614', 'rosR',
       'hlx2', 'cspD1', 'cspD2', 'sirR', 'VNG0194H', 'hrg']

	# put data in s x n shape, with s samples and n timepoints
	pivot = temp.unstack(-1)
	pivot.columns = [t for s,t in pivot.columns.values]

	effects = []
	selectStrain = pivot.index.get_level_values('Strain').isin(strains)
	selectCondition = pd.Series([False]*pivot.shape[0],index=pivot.index)
	if standard:
		selectCondition = selectCondition | (pivot.index.get_level_values('standard')==1)
		effects+=['standard']
	if paraquat:
		selectCondition = selectCondition | (pivot.index.get_level_values('paraquat')==1)
		effects+=['paraquat']
	if osmotic:
		selectCondition = selectCondition | (pivot.index.get_level_values('osmotic')==1)
		effects+=['osmotic']
	if heatshock:
		selectCondition = selectCondition | (pivot.index.get_level_values('heatshock')==1)
		effects+=['heatshock']
	select = selectStrain & selectCondition
	pivot = pivot.loc[select,:]

	fact,labels = pd.factorize(pivot.index.get_level_values('Strain'))
	e = []
	for eff in effects:
		e.append(pivot.index.get_level_values(eff))
	e = np.array(e).T
	e = np.where(e!=0)[1]
	e = np.array([fact,e]).T

	if len(effects) <= 1:
		e = e[:,0][:,None]

	x = pivot.columns.values[:,None]
	y = pivot.values.T

	return x,y,e,labels


def hsalinarum_data():
	import patsy

	data = pd.read_excel("data/hsalinarum/Raw_growth_data2.xlsx",sheetname='Raw data (OD600)SLIM')
	# time = np.arange(4,48,.5)
	time = np.arange(4,48,2)

	# temp = data[(data.Condition.isnull()) & ((data.Strain == 'ura3') | (data.Strain=='trmB') | (data.Strain == 'rosR'))]
	# temp = data[(data.Condition.isnull()) & ((data.Strain=='trmB') | (data.Strain == 'rosR'))]
	# temp = data[(data.Condition.isnull()) & ((data.Strain=='trmB') | (data.Strain == 'rosR') | (data.Strain == 'trh2'))]
	temp = data[(data.Condition.isnull()) & ((data.Strain=='trmB') | (data.Strain == 'rosR') | (data.Strain == 'trh2') | (data.Strain == 'idr1'))]
	# temp = data[(data.Condition.isnull()) & (data.Strain!='ura3')]

	y = temp[time].T.values
	y = np.log2(y)
	y = y - y[0,:]

	x = time
	x = (x-x.mean())/x.std()
	x = x[:,None]

	effect = patsy.dmatrix('C(Strain)+0',temp)
	effect = np.where(effect!=0)[1][:,None]

	# effect = (temp.Strain != "ura3").astype(int).values[:,None]

	return x,y,effect

def hsalinarum_pq_data():
	import patsy

	data = pd.read_excel("data/hsalinarum/Raw_growth_data2.xlsx",sheetname='Raw data (OD600)SLIM')
	# time = np.arange(4,48,.5)
	time = np.arange(4,48,2)

	# temp = data[(data.Condition.isnull()) & ((data.Strain == 'ura3') | (data.Strain=='trmB') | (data.Strain == 'rosR'))]
	# temp = data[(data.Condition.isnull()) & ((data.Strain=='trmB') | (data.Strain == 'rosR'))]
	# temp = data[(data.Condition.isnull()) & ((data.Strain=='trmB') | (data.Strain == 'rosR') | (data.Strain == 'trh2'))]
	temp = data[(data.Condition.isnull() | (data.Condition=='0.333mM PQ')) & ((data.Strain == 'rosR') | (data.Strain == 'idr1'))]
	# temp = data[(data.Condition.isnull()) & (data.Strain!='ura3')]

	temp.Condition[temp.Condition.isnull()] = ''

	y = temp[time].T.values
	y = np.log2(y)
	y = y - y[0,:]

	x = time
	x = (x-x.mean())/x.std()
	x = x[:,None]

	effect1 = patsy.dmatrix('C(Strain)+0',temp)
	effect1 = np.where(effect1!=0)[1]

	effect2 = patsy.dmatrix('C(Condition)+0',temp)
	effect2 = np.where(effect2!=0)[1]

	print effect1, effect2

	effect = np.column_stack((effect1,effect2))

	# effect = (temp.Strain != "ura3").astype(int).values[:,None]

	return x,y,effect

def hsalinarum_osmo_data(strain = None):
	import patsy

	if strain is None:
		strain = 'phoU'

	tidy = pd.read_csv("data/hsalinarum/tidy_normalize_all.csv")
	tidy['Experiment_Well'] = tidy.Experiment + "_" + tidy.Well.astype(str)
	pivot = tidy.pivot('Experiment_Well','time','OD')

	g = tidy.groupby(['Strain','standard','paraquat','peroxide','osmotic','heatshock'])
	ind = g.get_group(('ura3',1,0,0,0,0)).Experiment_Well.unique().tolist() + g.get_group(('ura3',0,0,0,1,0)).Experiment_Well.unique().tolist() + \
			g.get_group((strain,1,0,0,0,0)).Experiment_Well.unique().tolist() + g.get_group((strain,0,0,0,1,0)).Experiment_Well.unique().tolist()

	temp = pivot.loc[ind,:]

	time = np.arange(4,42,2)

	# data = pd.read_excel("data/hsalinarum/Raw_growth_data2.xlsx",sheetname='Raw data (OD600)SLIM')

	# temp = data[(data.Condition.isnull()) & ((data.Strain == 'ura3') | (data.Strain=='trmB') | (data.Strain == 'rosR'))]
	# temp = data[(data.Condition.isnull()) & ((data.Strain=='trmB') | (data.Strain == 'rosR'))]
	# temp = data[(data.Condition.isnull()) & ((data.Strain=='trmB') | (data.Strain == 'rosR') | (data.Strain == 'trh2'))]
	# temp = data[(data.Condition.isnull() | (data.Condition=='2.9M NaCl')) & ((data.Strain == 'trh2') | (data.Strain == 'idr1'))]
	# temp = data[(data.Condition.isnull()) & (data.Strain!='ura3')]

	# temp.Condition[temp.Condition.isnull()] = ''

	y = temp[time].T.values
	# y = np.log2(y)
	# y = y - y[0,:]

	x = time
	x = (x-x.mean())/x.std()
	x = x[:,None]

	# effect1 = patsy.dmatrix('C(Strain)+0',temp)
	# effect1 = np.where(effect1!=0)[1]
	#
	# effect2 = patsy.dmatrix('C(Condition)+0',temp)
	# effect2 = np.where(effect2!=0)[1]
	#
	# print effect1, effect2
	#
	# effect = np.column_stack((effect1,effect2))

	effect = np.repeat([[0,0]],g.get_group(('ura3',1,0,0,0,0)).Experiment_Well.unique().shape[0],0)
	effect = np.row_stack((effect,np.repeat([[0,1]],g.get_group(('ura3',0,0,0,1,0)).Experiment_Well.unique().shape[0],0)))
	effect = np.row_stack((effect,np.repeat([[1,0]],g.get_group((strain,1,0,0,0,0)).Experiment_Well.unique().shape[0],0)))
	effect = np.row_stack((effect,np.repeat([[1,1]],g.get_group((strain,0,0,0,1,0)).Experiment_Well.unique().shape[0],0)))

	#  [[0,0] * g.get_group(('ura3',1,0,0,0,0)).shape[0]] + [[0,1] * g.get_group(('ura3',0,1,0,0,0)).shape[0]] + \
	# 		[[1,0] * g.get_group((strain,1,0,0,0,0)).shape[0]] + [[1,1] * g.get_group((strain,0,1,0,0,0)).shape[0]]
	# effect = np.array(effect)

	return x,y,effect


def hsalinarum_strain_data(strains=['trmB'],time=None):
	import patsy

	tidy = pd.read_csv("data/hsalinarum/tidy_normalize_all.csv")
	tidy['Experiment_Well'] = tidy.Experiment + "_" + tidy.Well.astype(str)
	pivot = tidy.pivot('Experiment_Well','time','OD')

	g = tidy.groupby(['Strain','standard'])

	ind = []
	effect = []

	# for i,tup in enumerate([('ura3',1),('trmB',1)]):
	for i,tup in enumerate(zip(['ura3']+strains,[1]*(len(strains)+1))):
		temp = g.get_group(tup).Experiment_Well.unique().tolist()
		ind += temp
		effect += [i]*len(temp)
	# ind = g.get_group(('ura3',1)).Experiment_Well.unique().tolist() + g.get_group(('trmB',1)).Experiment_Well.unique().tolist()

	temp = pivot.loc[ind,:]

	if time is None:
		time = np.arange(4,42,2)

	y = temp[time].T.values

	x = time
	x = (x-x.mean())/x.std()
	x = x[:,None]

	effect = np.array(effect)[:,None]

	return x,y,effect

def hsalinarum_replicate_data():
	import patsy

	data = pd.read_excel("data/hsalinarum/Raw_growth_data2.xlsx",sheetname='Raw data (OD600)SLIM')
	# time = np.arange(4,48,.5)
	time = np.arange(4,48,4)

	# temp = data[(data.Condition.isnull()) & ((data.Strain == 'ura3') | (data.Strain=='trmB') | (data.Strain == 'rosR'))]
	# temp = data[(data.Condition.isnull()) & ((data.Strain=='trmB') | (data.Strain == 'rosR'))]
	# temp = data[(data.Condition.isnull()) & ((data.Strain=='trmB') | (data.Strain == 'rosR') | (data.Strain == 'trh2'))]
	temp = data[(data.Condition.isnull()) & ((data.Strain == 'ura3'))]
	# temp = data[(data.Condition.isnull()) & (data.Strain!='ura3')]

	temp.Condition[temp.Condition.isnull()] = ''

	y = temp[time].T.values
	y = np.log2(y)
	y = y - y[0,:]

	x = time
	x = (x-x.mean())/x.std()
	x = x[:,None]

	effect = patsy.dmatrix('C(Experiment):C(Well)+0',temp)
	effect = np.where(effect!=0)[1][:,None]

	return x,y,effect

def hsalinarum_beer_data(sample=1):
	data = pd.read_csv("data/hsalinarum/beer_et_al_2014/composite.csv",index_col=range(3))

	x = data.columns.values.astype(float);
	y = data.values
	effect = np.array(data.index.labels).T
	effect_map = [list(l) for l in list(data.index.levels)]

	select = x < 50
	x = x[select]
	y = y[:,select]

	select = np.arange(0,x.shape[0],sample)
	x = x[select]
	y = y[:,select]

	# select = effect[:,0] != 4
	# y = y[select,:]
	# effect = effect[select,:]

	x = (x-x.mean())/x.std()
	x = x[:,None]
	y = y.T

	return x,y,effect,effect_map

def plot_model_vs_true(m,mean,effect,interaction):

	nrows = 3
	ncols = sum([e.shape[1] for e in effect])

	plt.subplot(nrows,ncols,1)
	plt.plot(m.parameter_history[m.mu_index()].T.values,c='k',alpha=.1); plt.plot(mean,c='r');

	ind = 0
	for i in range(len(effect)):
		for j in range(effect[i].shape[1]):
			plt.subplot(nrows,ncols,ncols+1+ind)
			plt.plot(m.parameter_history[m.effectIndex(i,j)].T.values,c='k',alpha=.1); plt.plot(effect[i][:,j],c='r');
			plt.title("%d, %d" %(i,j))
			ind += 1

	ind = 0
	for i in range(interaction.shape[1]):
		for j in range(interaction.shape[2]):
			plt.subplot(nrows,ncols,2*ncols+1+ind)
			plt.plot(m.parameter_history[m.effectInteractionIndex(1,j,0,i)].T.values,c='k',alpha=.1); plt.plot(interaction[:,i,j],c='r');
			plt.title("%d, %d" %(i,j))
			ind += 1
