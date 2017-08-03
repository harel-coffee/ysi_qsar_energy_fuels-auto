#!/bin/bash
#$ -V

#PBS -N mongo_master
#PBS -V
#PBS -o _mongo_master.out
#PBS -e	_mongo_master.err

source ~/.bash_profile
cd $PBS_O_WORKDIR

/home/pstjohn/anaconda/envs/py3k/bin/python3 /home/pstjohn/pstjohn/Projects/YSI/qspr_model_creation/hyperopt/keras_start.py
