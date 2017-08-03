import numpy as np
np.random.seed(0)

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (StandardScaler, Normalizer, RobustScaler,
                                   MaxAbsScaler, Imputer)
from sklearn.decomposition import PCA

from sklearn.feature_selection import VarianceThreshold, RFE, SelectKBest, f_regression
from sklearn import svm

from ysi_utils.descriptors import dragon
from ysi_utils.validation import y_train as y
from ysi_utils.validation import y_test

X = dragon.loc[y.index]
X_test = dragon.loc[y_test.index]

# These are the optimal parameters as selected by the hyperopt fitting routine.
params = \
{'activation': 'relu',
 'batch_size': 39,
 'dropout1': 0.36632971139053216,
 'dropout2': 0.030115015871571206,
 'feature_extractor': {'n_features_to_select': 390, 'step': 10, 'type': 'RFE'},
 'nb_epochs': 100,
 'num_layers': {'dropout3': 0.31435729897990794,
                'layers': 'three',
                'units3': 161},
 'optimizer': 'adam',
 'preprocessor': 'MaxAbsScaler',
 'units1': 140,
 'units2': 140}

def get_input_size(params):
    if 'n_components' in params['feature_extractor']:
        return params['feature_extractor']['n_components']

    elif 'k' in params['feature_extractor']:
        return params['feature_extractor']['k']

    elif 'n_features_to_select' in params['feature_extractor']:
        return params['feature_extractor']['n_features_to_select']

def build_model(params=params):
    model = Sequential()
    model.add(Dense(output_dim=params['units1'], 
                    input_dim=get_input_size(params))) 
    model.add(Activation(params['activation']))
    model.add(Dropout(params['dropout1']))

    model.add(Dense(output_dim=params['units2'], init="glorot_uniform"))
    model.add(Activation(params['activation']))
    model.add(Dropout(params['dropout2']))

    if params['num_layers']['layers']== 'three':
        model.add(Dense(output_dim=params['num_layers']['units3'], init="glorot_uniform"))
        model.add(Activation(params['activation']))
        model.add(Dropout(params['num_layers']['dropout3']))    

    model.add(Dense(output_dim=1))
    model.compile(loss='mae', optimizer=params['optimizer'])

    return model


ffann = KerasRegressor(build_fn=build_model,
                       nb_epoch=params['nb_epochs'],
                       batch_size=params['batch_size'],
                       verbose=0)

# Set up preprocessing pipeline
imputer = Imputer()
var_filter = VarianceThreshold()

preprocessor_dict = {
    'StandardScaler' : StandardScaler,
    'MaxAbsScaler' : MaxAbsScaler,
    'Normalizer' : Normalizer,
    'RobustScaler' : RobustScaler,
}

scaler = preprocessor_dict[params['preprocessor']]()

if params['feature_extractor']['type'] == 'pca':
    opts = dict(params['feature_extractor'])
    del opts['type']
    feature_extraction = PCA(**opts)

elif params['feature_extractor']['type'] == 'RFE':
    opts = dict(params['feature_extractor'])
    del opts['type']
    svr = svm.SVR(kernel='linear')
    feature_extraction = RFE(estimator=svr, **opts)

elif params['feature_extractor']['type'] == 'SelectKBest':
    opts = dict(params['feature_extractor'])
    del opts['type']
    feature_extraction = SelectKBest(score_func=f_regression, **opts)

model = Pipeline(steps=[
    ('imputer', imputer),
    ('filter', var_filter),
    ('scaler', scaler),
    ('feature_extraction', feature_extraction),
    ('ffann', ffann)
])
