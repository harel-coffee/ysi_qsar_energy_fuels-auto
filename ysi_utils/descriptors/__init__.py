import os
import pandas as pd

currdir = os.path.dirname(os.path.abspath(__file__))

# low = pd.read_pickle(currdir + '/low.p')
# high = pd.read_pickle(currdir + '/high.p')

for path in os.listdir(currdir):
    filename, extension = os.path.splitext(path)
    if extension == '.p':
        locals().update(
            {filename : pd.read_pickle(currdir + '/' + path)})
