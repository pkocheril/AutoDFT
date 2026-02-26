#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=167:59:59   # walltime; max. 168 hours
#SBATCH --ntasks=32   # number of processor cores (i.e. tasks); max. 32
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=6G   # memory per CPU core; max 6 GB/core (192 GB total)
#SBATCH -J "AutoDFT"   # job name
#SBATCH --mail-user=$USER@caltech.edu   # email address; update as needed
#SBATCH --mail-type=ALL
#SBATCH --constraint=skylake|cascadelake|broadwell

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE

### Combined AutoDFT + FCclasses
module load mkl
python auto_DFT.py --cores $(nproc) --mem $(nmem)

