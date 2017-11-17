from data import get_train_data
from os import path
import joblib

CACHE_FILENAME = 'cached_data.pickle'

def load_data(use_cached=True):
    if use_cached and path.isfile(CACHE_FILENAME):
        data = joblib.load(open(CACHE_FILENAME, 'rb'))
        X_train = data['X_train']
        y_train = data['y_train']
    else:
        X_train, y_train = get_train_data()
        data = {
            'X_train': X_train,
            'y_train': y_train
        }
        joblib.dump(data, open(CACHE_FILENAME, 'wb'), compress=True)
    return X_train, y_train