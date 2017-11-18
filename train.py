from data import get_train_data
from os import path
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential, load_model
from keras.utils import to_categorical
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras import regularizers
import joblib

CACHE_FILENAME = 'cached_data.pickle'

def load_data(use_cached=True):
    if use_cached and path.isfile(CACHE_FILENAME):
        data = joblib.load(open(CACHE_FILENAME, 'rb'))
        X_train = data['X_train']
        y_train = data['y_train']
        vectorizer = data['vectorizer']
    else:
        X_train, y_train, vectorizer = get_train_data()
        data = {
            'X_train': X_train,
            'y_train': y_train,
            'vectorizer': vectorizer
        }
        joblib.dump(data, open(CACHE_FILENAME, 'wb'), compress=True)
    return X_train, y_train, vectorizer

def build_model():
    X_train, y_train, vectorizer = load_data()
    
    model = Sequential()
    model.add(LSTM(256, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(vectorizer.dim, 
        activation='linear',
        kernel_regularizer=regularizers.l2(0.01), 
        activity_regularizer=regularizers.l1(0.01)))
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    FILE_PATH='weights-{epoch:02d}-{loss:.4f}.h5'
    callbacks = [
        TensorBoard(log_dir='./logs'),
        ModelCheckpoint(FILE_PATH, 
            monitor='loss', 
            verbose=1, 
            save_best_only=True, 
            mode='min',
            period=20)
    ]

    model.fit(X_train, y_train, epochs=500, batch_size=16, callbacks=callbacks)

    model.save('final_model.h5')
    return model
