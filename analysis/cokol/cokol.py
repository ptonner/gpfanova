import os
import pandas as pd

data_dir = "data/cokol_et_al_2011/"
data_files = os.listdir(data_dir)
data_files.remove("README.txt")
# data_files.remove(".DS_Store")

def fileNameParse(df):
	import re
	a1,a2 = re.match('([A-Za-z0-9]*)-([A-Za-z0-9]*)\.txt',df).groups()
	return a1,a2

def listAntibiotics():

	a = []
	for df in data_files:
		a1,a2 = fileNameParse(df)
		if not a1 in a:
			a.append(a1)
		if not a2 in a:
			a.append(a2)
	return a

def hasSelfDataset(a):
	data_files = os.listdir(data_dir)
	return '%s-%s.txt' % (a,a) in data_files

def column_concentrations(ind):
    return 1./8*(ind/8),1.*ind%8/8

def load(a1,a2,t0=0,step=1,thin=True):
	import numpy as np

	data = pd.read_csv(os.path.join(data_dir,'%s-%s.txt'%(a1,a2)),sep="\t",header=None)
	data = data.iloc[:,:-1]

	select = data.index > t0
	data = data.loc[select,:]

	data = np.log2(data)
	# data = data.iloc[4:,:]
	data = data - data.iloc[0,:]

	x = data.index.values[:,None]
	x = (x.astype(float)-x.min())/x.max()

	y = data.values

	# select = x[:,0]>t0
	# x = x[select,:]
	# y = y[select,:]

	concs = [column_concentrations(i) for i in data.columns]
	concs = pd.DataFrame(concs)
	effect = np.array([concs[0].factorize()[0],concs[1].factorize()[0]]).T
	labels = [concs[0].tolist(),concs[1].tolist()]

	if thin:
		select = np.arange(0,effect.shape[0],step)
		select = np.all(effect % 2 == 0,1)
		y = y[:,select]
		effect = effect[select,:]
		effect = effect/step

	return x,y,effect,labels

def loadModel(a1,a2,**kwargs):
	import gpfanova

	x,y,effect,labels = load(a1,a2,**kwargs)
	m = gpfanova.fanova.FANOVA(x,y,effect,interactions=True,parameter_file='results/cokol/%s-%s_interactions.csv'%(a1,a2),helmert_convert=True)
	return m,x,y,effect,labels

def generate_commands(n=10,interactions=False):

	ret = []
	s = ''
	if interactions:
		s = '-i '
	for f in data_files:
		a1,a2 = fileNameParse(f)

		ret.append('sbatch cokol.sh %s-n %d %s %s' % (s,n,a1,a2))

	return ret

def analyze():
	import os,re
	import gpfanova
	import matplotlib.pyplot as plt

	resultsDir = "results/cokol"
	outputDir = os.path.join(resultsDir,'figures')
	results = os.listdir(resultsDir)
	results.remove('figures')

	for res in results:
		match = re.match("([0-9a-zA-Z]+)-([0-9a-zA-Z]+)_interactions.csv",res)
		if not match:
			continue
		a1,a2 = match.groups()
		print a1,a2
		m,_,_,_,_ = loadModel(a1,a2,t0=6,thin=False)

		pair_dir = os.path.join(outputDir,'%s_%s'%(a1,a2))
		if not '%s_%s'%(a1,a2) in os.listdir(outputDir):
			os.mkdir(pair_dir)


		plt.figure(figsize=(16,8))
		plt.subplot(121)
		plt.title(a1,fontsize=20)
		gpfanova.plot.plotSingleEffect(m,0,function=True,offset=False,alpha=.01,variance=False,burnin=0,origin=True,_mean=False);
		plt.subplot(122)
		plt.title(a2,fontsize=20)
		gpfanova.plot.plotSingleEffect(m,1,function=True,offset=False,alpha=.01,variance=False,burnin=0,origin=True,_mean=False);
		plt.tight_layout()
		plt.savefig(os.path.join(pair_dir,'singleEffectFunctions.png'))
		plt.close()

		plt.figure(figsize=(16,8))
		plt.subplot(121)
		plt.title(a1,fontsize=20)
		gpfanova.plot.plotSingleEffect(m,0,data=True,individual=True,empirical=True);
		plt.subplot(122)
		plt.title(a2,fontsize=20)
		gpfanova.plot.plotSingleEffect(m,1,data=True,individual=True,empirical=True);
		plt.tight_layout()
		plt.savefig(os.path.join(pair_dir,'singleEffectData.png'))
		plt.close()

		plt.figure(figsize=(20,20))
		gpfanova.plot.plotInteraction(m,0,1,function=True,subplots=True,origin=True,offset=False,relative=True);
		plt.tight_layout()
		plt.savefig(os.path.join(pair_dir,'interactionRelativeFixed.png'))
		plt.close()

		plt.figure(figsize=(20,20))
		gpfanova.plot.plotInteraction(m,0,1,function=True,subplots=True,origin=True,offset=False,relative=True,controlFixed=False);
		plt.tight_layout()
		plt.savefig(os.path.join(pair_dir,'interactionRelativeDynamic.png'))
		plt.close()

		plt.figure(figsize=(20,20))
		gpfanova.plot.plotInteraction(m,0,1,function=True,subplots=True,origin=True,offset=False,relative=False);
		plt.tight_layout()
		plt.savefig(os.path.join(pair_dir,'interactionActual.png'))
		plt.close()


if __name__ == "__main__":
	import argparse, gpfanova

	parser = argparse.ArgumentParser(description='Run analysis of Cokol et al. data.')
	parser.add_argument('antibiotics',metavar=('A'), type=str, nargs='*',
	                  help='antibiotics to build model for')
	parser.add_argument('-n', dest='n_samples', action='store',default=10, type=int,
	                   help='number of samples to generate from posterior')
	parser.add_argument('-i', dest='interactions', action='store_true',
	                   help='include interactions in the model')
	parser.add_argument('-g', dest='generateCommands', action='store_true',
	                   help='generate the commands for this script')
	parser.add_argument('-a', dest='analyze', action='store_true',
	                   help='analyze the output of this script')

	args = parser.parse_args()

	if args.generateCommands:
		print '\n'.join(generate_commands(args.n_samples,args.interactions))
	elif args.analyze:
		analyze()
	else:
		a1,a2 = args.antibiotics[0],args.antibiotics[1]
		x,y,effect,_ = load(a1,a2,t0=6,step=2)
		m = gpfanova.fanova.FANOVA(x,y,effect,interactions=args.interactions,helmert_covert=True)

		s = ''
		if args.interactions:
			s = '_interactions'

		try:
			m.sample(args.n_samples,10)
		except Exception,e:
			m.save('results/cokol/%s-%s%s.csv'%(a1,a2,s))
			raise(e)

		m.save('results/cokol/%s-%s%s.csv'%(a1,a2,s))
