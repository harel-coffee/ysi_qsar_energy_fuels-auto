import os

import numpy as np
from sklearn.externals import joblib

currdir = os.path.dirname(os.path.abspath(__file__))
outlier_model = joblib.load(currdir + '/outliers.pkl')
bagging_model = lambda: joblib.load(currdir + '/bagging_model.pkl.large')

def ensemble_predict(X):
    return np.hstack([estim.predict(X) for estim in
                      ensemble_predict.bagging_model.estimators_])
