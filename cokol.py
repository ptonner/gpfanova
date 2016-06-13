import os
import pandas as pd

data_dir = "data/cokol_et_al_2011/"
data_files = os.listdir(data_dir)
data_files.remove("README.txt")
data_files.remove(".DS_Store")

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
    return 1.*ind/64,1.*ind%8/8

def load(a1,a2):

	data = pd.read_csv(os.path.join(data_dir,'%s-%s.txt'%(a1,a2)),sep="\t",header=None)
	x = data.index.values
	y = data.values

	concs = [column_concentrations(i) for i in data.columns]
	concs = pd.DataFrame(concs)
	effect = [concs[0].factorize()[0],concs[0].factorize()[1]]
	labels = [concs[0].tolist(),concs[1].tolist()]

	return x,y,effect,labels

def generate_commands(n=10,interactions=False):

	ret = []
	s = ''
	if interactions:
		s = '-i '
	for f in data_files:
		a1,a2 = fileNameParse(f)

		ret.append('sbatch cokol.sh %s-n %d %s %s' % (s,n,a1,a2))

	return ret

if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description='Run analysis of Cokol et al. data.')
	parser.add_argument('antibiotics',metavar=('A'), type=str, nargs='*',
	                  help='antibiotics to build model for')
	parser.add_argument('-n', dest='n_samples', action='store',default=10, type=int,
	                   help='number of samples to generate from posterior')
	parser.add_argument('-i', dest='interactions', action='store_true',
	                   help='include interactions in the model')
	parser.add_argument('-g', dest='generateCommands', action='store_true',
	                   help='generate the commands for this script')

	args = parser.parse_args()

	if args.generateCommands:
		print '\n'.join(generate_commands(args.n_samples,args.interactions))
	else:
		print args
