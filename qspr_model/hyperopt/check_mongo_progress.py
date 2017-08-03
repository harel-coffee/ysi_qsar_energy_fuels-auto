#!/home/pstjohn/anaconda/envs/py3k/bin/python

import pprint
from hyperopt.mongoexp import MongoTrials
from collections import Counter

trials = MongoTrials('mongo://skynet.hpc.nrel.gov:1234/keras_db2/jobs',
                     exp_key='715_keras_qm')


status = Counter((t['status'] for t in trials.results))

if 'ok' not in status: status['ok'] = 0

print("Number of in-progress trials: {}".format(status['new']))
print("Number of trial results: {}".format(status['ok']))

if status['ok'] > 0:
    print("Best loss: {:.2f}".format(trials.best_trial['result']['loss']))
    print("Optimal Parameters:")

    pp = pprint.PrettyPrinter()
    pp.pprint(trials.best_trial['result']['params'].to_dict())

def save_progess(trials):
    import pickle
    with open('trials.p', 'wb') as f:
        pickle.dump(trials.trials, f)

    with open('results.p', 'wb') as f:
        pickle.dump(trials.results, f)
