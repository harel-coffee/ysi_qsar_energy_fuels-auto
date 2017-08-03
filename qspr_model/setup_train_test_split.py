import numpy as np
from ysi_utils.data import low

rstate = np.random.RandomState(0)
low.set_index('SMILES', inplace=True)
y_test_low = low.YSI.sample(frac=.2, random_state=rstate)
y_train_low = low[~low.index.isin(y_test_low.index)]

low.loc[y_test_low.index].to_csv('../ysi_utils/validation/y_test.csv')
low.loc[y_train_low.index].to_csv('../ysi_utils/validation/y_train.csv')
