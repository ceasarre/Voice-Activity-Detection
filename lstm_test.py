from sklearn.metrics import confusion_matrix
from objective_params import analyze_objective_params
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers import Dropout
from sklearn.preprocessing import StandardScaler
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import json
import numpy as np
import csv
import pandas as pd
import pickle
import multiprocessing
import os

DATA_TRAIN = 'train.json'
DATA_EVAL = 'eval_rand.json'
DATA_TEST = 'test_stream.json'
NAME_DEV = "radiokom"
PATH_TO_MODEL = 'lstm_res\tested_4\lstm_100_lstm_100_dense_100\lstm_100_lstm_100_dense_100.h5'

def create_model(neurons_1=2, neurons_2 = 2, neurons_3 = 4):
    # create model
    model = Sequential()
    model.add(LSTM(neurons_1, input_shape=(8000, 1), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(neurons_2, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(neurons_3, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(int(neurons_3/2), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = Adam(learning_rate=1e-4)
    # Compile model
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer, metrics=['accuracy'])

    return model


def load_data(data_path):

    with open(data_path, "r") as fp:
        data = json.loads(fp.read())

    X = np.array(data["samples"])
    y = np.array(data["labels"])

    return X, y

# plot VAD's decision
def plot_res(y, y_pred):
    
    y = y[1080:1180]
    y_pred = y_pred[1080:1180]
    
    plt.style.use('ggplot')
    fig, ax = plt.subplots()
    ax.set_title('wykrywanie aktywności głosowej za pomocą sieci LSTM')
    ax.set_xlabel('numer ramki')
    ax.set_ylabel('mowa/brak mowy')
    ax.plot(y,color = 'b', label = 'stan faktyczny')
    ax.plot(y_pred,color = 'r', linestyle = '--', label = 'decyzja algorytmu')
    ax.set_yticks([0,1])
    ax.set_yticklabels(['brak mowy', 'mowa'])
    ax.legend(loc ="lower right")
    fig.savefig('plots_final\lstm.png')
    # plt.show()

def prepare_dataset(data_train, data_eval, data_test):

    # create train and validation sets
    X_train, y_train = load_data(data_train)

    X_validation, y_validation = load_data(data_eval)

    X_test, y_test = load_data(data_test)

    # ? Feature scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_validation = sc.transform(X_validation)
    X_test = sc.transform((X_test))

    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test



if __name__ == "__main__":
    # load dataset
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_dataset(
        DATA_TRAIN, DATA_EVAL, DATA_TEST)
    
    model = keras.models.load_model(PATH_TO_MODEL)
    y_pred_val = model.predict(X_test)
    
    # treshold
    y_pred_val = [0 if x < 0.5 else 1 for x in y_pred_val]

    cm = confusion_matrix(y_test, y_pred_val)
    print(cm)

    signal, noise, fec, msc, over, nds = analyze_objective_params(
        y_test, y_pred_val)


    print("signal {}, noise {}, fec {}, msc {}, over {}, nds {}".format(
        signal, noise, fec, msc, over, nds))

    plot_res(y_test, y_pred_val)
    