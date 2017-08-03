import os
import pandas as pd

currdir = os.path.dirname(os.path.abspath(__file__))

y_train = pd.read_csv(currdir + '/y_train.csv', index_col=0)
y_test = pd.read_csv(currdir + '/y_test.csv', index_col=0)
