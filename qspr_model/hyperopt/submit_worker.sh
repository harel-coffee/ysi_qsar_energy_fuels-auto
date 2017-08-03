#PBS -l nodes=1:ppn=1
#PBS -N mongo_worker
#PBS -V
#PBS -o _mongo_worker.out
#PBS -e	_mongo_worker.err

source ~/.bash_profile
cd $TMPDIR

/home/pstjohn/anaconda/envs/py3k/bin/hyperopt-mongo-worker --mongo=skynet.hpc.nrel.gov:1234/keras_db2 --workdir=$TMPDIR
