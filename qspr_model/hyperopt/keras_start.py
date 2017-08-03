from functools import partial

from hyperopt import fmin, tpe, hp
from hyperopt.mongoexp import MongoTrials

space = {'num_layers': hp.choice(
             'num_layers', [
                 {
                     'layers':'two',
                 }, {
                     'layers':'three',
                     'units3': hp.quniform_int('units3', 1, 200, 1), 
                     'dropout3': hp.uniform('dropout3', 0.,.75)
                 }]),

         'units1': hp.quniform_int('units1', 1, 200, 10),
         'units2': hp.quniform_int('units2', 1, 200, 10),

         'dropout1': hp.uniform('dropout1', 0.,.75),
         'dropout2': hp.uniform('dropout2', 0.,.75),

         'batch_size' : hp.quniform_int('batch_size', 28, 128, 1),

         'nb_epochs' : 100,
         'optimizer': hp.choice('optimizer',['adadelta', 'adam', 'rmsprop', 'adamax']),
         'activation': 'relu',

         'preprocessor' : hp.choice(
             'preprocessor', ['StandardScaler', 'RobustScaler', 'Normalizer', 'MaxAbsScaler']),

         'feature_extractor' : hp.choice(
             'feature_extractor', [
                 {
                     'type' : 'pca',
                     'n_components': hp.quniform_int('n_components', 10, 190, 10),
                     'whiten': hp.choice('whiten', [True, False])
                 }, {
                     'type' : 'RFE',
                     'n_features_to_select' : hp.quniform_int(
                         'n_features_to_select', 10, 500, 10),
                     'step' : 10,
                 }, {
                     'type' : 'SelectKBest',
                     'k' : hp.quniform_int('k', 10, 500, 10),
                 }]),
         }

def f_nn(params, X, y):   

    import numpy as np
    np.random.seed(0)

    from hyperopt import STATUS_OK

    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation
    from keras.wrappers.scikit_learn import KerasRegressor

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import (StandardScaler, Normalizer, RobustScaler,
                                       MaxAbsScaler, Imputer, LabelEncoder)
    from sklearn.decomposition import PCA
    from sklearn.metrics import median_absolute_error
    from sklearn.cross_validation import cross_val_predict, LeaveOneLabelOut

    from sklearn.feature_selection import VarianceThreshold, RFE, SelectKBest, f_regression
    from sklearn import svm

    print('Params testing: ', params)

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
    
    labels = LabelEncoder().fit_transform(y.Type)

    n_repeats = 5
    acc = np.zeros(n_repeats)

    for i in range(n_repeats):

        y_cv_predict = cross_val_predict(model, X.values, y.YSI.values,
                                              cv=LeaveOneLabelOut(labels))
        acc[i] = median_absolute_error(y.YSI.values, y_cv_predict)

    return {'loss': acc.mean(),
            'loss_variance': acc.std()**2,
            'status': STATUS_OK,
            'params': params,
            'y_pred': y_cv_predict} 


from ysi_utils.descriptors import dragon_qm as dragon
from ysi_utils.qspr import y_train as y
X = dragon.loc[y.index]

trials = MongoTrials('mongo://skynet.hpc.nrel.gov:1234/keras_db2/jobs', exp_key='715_keras_qm')
# from hyperopt.mongoexp import Trials
# trials = Trials()

best = fmin(
    fn=partial(f_nn, X=X, y=y),
    space=space,
    algo=tpe.suggest,
    max_evals=3000,
    trials=trials)

print(best)
