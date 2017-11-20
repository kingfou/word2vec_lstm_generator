from data import get_train_data, window, get_data
from os import path
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential, load_model
from keras.utils import to_categorical
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras import regularizers
import numpy as np
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

def build_model(from_file=None):
    X_train, y_train, vectorizer = load_data()
    
    if from_file:
        model = load_model(from_file)
    else:
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
            save_best_only=False, 
            mode='min',
            period=20)
    ]

    model.fit(X_train, y_train, epochs=500, batch_size=16, callbacks=callbacks)

    model.save('final_model.h5')
    return model

CACHED_DATA = None
def predict(filename, num_chars=10, load_cached=True):
    if not num_chars:
        return ''
    if not CACHED_DATA or not load_cached:
        CACHED_DATA = load_data()
    X_train = CACHED_DATA['X_train']
    y_train = CACHED_DATA['y_train']
    vectorizer = CACHED_DATA['vectorizer']

    idx = np.random.randint(0, X_train.shape[0], size=1)[0]
    seed = X_train[idx]

    #Build model and load weights
    model = Sequential()
    model.add(LSTM(256, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(vectorizer.dim, 
        activation='linear',
        kernel_regularizer=regularizers.l2(0.01), 
        activity_regularizer=regularizers.l1(0.01)))
    model.load_weights(filename)
    model.compile(loss='mean_squared_error', optimizer='adam')

    # start generating
    def predict_transform(X, X_train):
        X_test = []
        y_test = []

        data = [X]
        data = get_data(data)
        data = [window(file_content, X_train.shape[1]) for file_content in data]

        for item in data:
            for X, y in item:
                X_point = vectorizer.transform(X)
                y_point = vectorizer.transform_single(y[0])
                X_test.append(X_point)
                y_test.append(y_point)

        X_test = np.array(X_test)
        y_test = np.array(y_test)

        return (X_test, y_test)

    def predict_inverse_transform(X, vectorizer):
        data = X[0]
        return vectorizer.inverse_transform_single(data)

    first_word = model.predict(seed)[0]
    first_word = predict_inverse_transform(first_word, vectorizer)
    result = first_word
    for idx in range(num_chars):
        aux_result = predict_transform(result, X_train)
        aux_result = model.predict(aux_result)
        aux_result = predict_inverse_transform(aux_result, vectorizer)
        result += ' ' + aux_result
    
    return result
