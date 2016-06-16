#!/bin/bash
#SBATCH --job-name=cokol
#SBATCH --output=cokol.out
#SBATCH --mem=4000

cd /dscrhome/pt59/dev/gp_fanova
source bin/activate
python cokol.py $1 $2 $3 $4 $5 $6
