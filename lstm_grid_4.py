import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
from objective_params import analyze_objective_params
from sklearn.preprocessing import StandardScaler
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
NAME_DEV = "DTX"

def create_model(neurons_1=2, neurons_2 = 2, neurons_3 = 4):
    # create model
    model = Sequential()
    model.add(layers.LSTM(neurons_1, input_shape=(8000, 1), return_sequences=True))
    model.add(layers.Dropout(0.2))
    model.add(layers.LSTM(neurons_2, return_sequences=False))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(neurons_3, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(int(neurons_3/2), activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
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

def prepare_dataset(data_train, data_eval, data_test):

    # create train and validation sets
    X_train, y_train = load_data(data_train)

    # X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size = validation_size)

    X_validation, y_validation = load_data(data_eval)

    X_test, y_test = load_data(data_test)

    # ? Feature scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_validation = sc.transform(X_validation)
    X_test = sc.transform((X_test))

    # ! TO TEST
    # X_train = X_train[0:10]
    # X_validation = X_validation[0:10]
    # X_test = X_test[0:10]

    # y_train = y_train[0:10]
    # y_validation = y_validation[0:10]
    # y_test = y_test[0:10]

    # add new axis to have right shape
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def plot_history(history, name, dir):

    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["acc"],
                label="Dokładność danych treningowych")
    axs[0].plot(history.history["val_acc"],
                label="Dokładność danych testowych")
    axs[0].set_ylabel("Dokładność")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Wykres dokladnosci od iteracji")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="błąd danych treningowych")
    axs[1].plot(history.history["val_loss"], label="błąd danych testowych")
    axs[1].set_ylabel("Błąd")
    axs[1].set_xlabel("iteracja")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Wykres błędu od iteracji")

    plt.savefig("{}/{}.png".format(dir,name))


if __name__ == "__main__":
    # load dataset
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_dataset(
        DATA_TRAIN, DATA_EVAL, DATA_TEST)
    
    grid_to_search = []

    # Grid params
    grid_1 = [20,20,20]
    grid_to_search.append(grid_1)
    grid_2 = [20,20,60]
    grid_to_search.append(grid_2)
    grid_3 = [20,20,100]
    grid_to_search.append(grid_3)
    grid_4 = [20,60,20]
    grid_to_search.append(grid_4)
    grid_5 = [20,60,60]
    grid_to_search.append(grid_5)
    grid_6 = [20,60,100]
    grid_to_search.append(grid_6)
    grid_7 = [20,100,20]
    grid_to_search.append(grid_7)
    grid_8 = [20,100,60]
    grid_to_search.append(grid_8)
    grid_9 = [20,100,100]
    grid_to_search.append(grid_9)

    grid_10 = [100,100,100]
    grid_to_search.append(grid_10)
    grid_11 = [100,100,60]
    grid_to_search.append(grid_11)
    grid_12 = [100,100,20]
    grid_to_search.append(grid_12)
    grid_13 = [100,60,100]
    grid_to_search.append(grid_13)
    grid_14 = [100,60,60]
    grid_to_search.append(grid_14)
    grid_15 = [100,60,20]
    grid_to_search.append(grid_15)
    grid_16 = [100,20,100]
    grid_to_search.append(grid_16)
    grid_17 = [100,20,60]
    grid_to_search.append(grid_17)
    grid_18 = [100,20,20]
    grid_to_search.append(grid_18)
    grid_19 = [60,100,100]
    grid_to_search.append(grid_19)
    grid_20 = [60,100,60]
    grid_to_search.append(grid_20)
    grid_21 = [60,100,20]
    grid_to_search.append(grid_21)
    grid_22 = [60,60,100]
    grid_to_search.append(grid_22)
    grid_23 = [60,60,60]
    grid_to_search.append(grid_23)
    grid_24 = [60,60,20]
    grid_to_search.append(grid_24)
    grid_25 = [60,20,100]
    grid_to_search.append(grid_25)
    grid_26 = [60,20,60]
    grid_to_search.append(grid_26)
    grid_27 = [60,20,20]
    grid_to_search.append(grid_27)

    
    for i, grid in enumerate(grid_to_search):
        path = "lstm_{}_lstm_{}_dense_{}".format(grid[0],grid[1],grid[2])
        os.makedirs(path)
        name = "lstm_res\tested_4\lstm_{}_lstm_{}_dense_{}".format(grid[0],grid[1],grid[2])
        model = create_model(grid[0],grid[1],grid[2])
        model.summary()

        print(r"model nr {}\{}".format(i+1,len(grid_to_search)))
        history = model.fit(X_train, y_train, validation_data=(
            X_validation, y_validation), batch_size=128, epochs=20)
       
        print("Saving model")
        model.save("{}/{}.h5".format(path,name))

        print("Saving history")
        plot_history(history,name,path)

        # save weigths
        with open("{}/{}_weights".format(path,name), 'wb') as file_pi:
            print("Saving weihts")
            pickle.dump(history.history, file_pi)

        # test model on validation
        print("Making prediction validation set")
        y_pred_val = model.predict(X_validation)
        # treshold
        y_pred_val = [0 if x < 0.5 else 1 for x in y_pred_val]

        # Confussion matrix validation data
        cm = confusion_matrix(y_validation, y_pred_val)
        accuracy_validation = 0
        print(cm)
        print("Validation accuracy calculating")
        accuracy_validation = ((cm[0][0] + cm[1][1])/(cm[0][0]+cm[1][1]+cm[1][0]+cm[0][1]))
        with open('{}/conf_val.txt'.format(path), 'w') as file:
            file.write("True Neg {}, False Pos {}, False neg {}, True Pos {}".format(cm[0][0], cm[0][1], cm[1][0], cm[1][1]))
      