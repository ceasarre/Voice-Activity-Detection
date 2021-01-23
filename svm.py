from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import json
from frame import Frame
from objective_params import analyze_objective_params
import matplotlib.pylab as pylab

# Params to plot
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

# Paths to data
DATA_TRAIN_FEATURES = 'train_features.csv'
DATA_TEST_FEATURES = 'eval_features.csv'
DATA_STREAM = 'test_stream.json'
DATA_TRAIN_SAMPLES = 'train.json'
DATA_TEST_SAMPLES = 'eval_rand.json'


# Load data
def load_data_features(data_path):

    dataset = pd.read_csv(data_path)

    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    return X, y


# Function do classify by SVM based on ZCR and STE, 
#   used only as tested feature
def classify_by_feature(X_train, X_test, y_train, y_test):

    # scale the features
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # classifier and fitting
    classifier = SVC(random_state=0, kernel='linear')
    classifier.fit(X_train, y_train)

    # predictions
    y_pred = classifier.predict(X_test)

    # confusion matrix

    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)

    return classifier, sc


# load raw data
def load_data_samples(data_path):

    with open(data_path) as fp:
        data = json.load(fp)

    X = np.array(data["samples"])
    y = np.array(data["labels"])
    mapping = np.array(data["mapping"])

    return X, y, mapping


# analyze by SVM, parameters found in gridsearch
def analyze_signal_samples(train_data, test_data):

    X_train, y_train, mapping = load_data_samples(train_data)
    X_test, y_test, mapping = load_data_samples(test_data)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    classifier_samples = SVC(random_state=0, kernel='rbf', verbose=1,C=1,gamma=0.0001)
    classifier_samples.fit(X_train, y_train)

    y_pred = classifier_samples.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)

    return y_test, y_pred


# plot of the train data
def visualize_dataset(x, y):

    plt.style.use('ggplot')

    X_set, y_set = x, y

    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], s=2,
                    c=ListedColormap(('red', 'green'))(i), label=j)

    # plt.title('Rozkład parametrów STE i ZCR danych treningowych')
    plt.xlabel('STE')
    plt.ylabel('ZCR')
    plt.legend()
    plt.show()

# plot VAD's decisions
def plot_res(y, y_pred):
    
    y = y[1080:1180]
    y_pred = y_pred[1080:1180]
    plt.rcParams.update({'font.size': 22})
    plt.style.use('ggplot')
    fig, ax = plt.subplots()
    ax.set_title('wykrywanie aktywności głosowej metodą SVM na wycinku sygnału odebranego')
    ax.set_xlabel('numer ramki')
    ax.set_ylabel('mowa/brak mowy')
    ax.plot(y,color = 'b', label = 'stan faktyczny')
    ax.plot(y_pred,color = 'r', linestyle = '--', label = 'decyzja algorytmu')
    ax.set_yticks([0,1])
    ax.set_yticklabels(['brak mowy', 'mowa'])
    ax.legend(loc ="lower right")
    fig.savefig('plots_final/svm.png')
    plt.show()


if __name__ == "__main__":

    # analyze validation data
    y_val, y_pred_val = analyze_signal_samples(
        DATA_TRAIN_SAMPLES, DATA_TEST_SAMPLES)

    # analyze test data
    y_test, y_pred = analyze_signal_samples(
        DATA_TRAIN_SAMPLES, DATA_STREAM)

    # objective params for test data
    signal, noise, fec, msc, over, nds = analyze_objective_params(
        y_test, y_pred)

    # objective params for validation data
    signal_val, noise_val, fec_val, msc_val, over_val, nds_val = analyze_objective_params(
        y_val, y_pred_val)

    print("signal {}, noise {}, fec {}, msc {}, over {}, nds {}".format(
        signal, noise, fec, msc, over, nds))
    with open('svm/objective_stream.txt', 'w') as file:
        file.write("signal {}, noise {}, fec {}, msc {}, over {}, nds {}".format(
            signal, noise, fec, msc, over, nds))

    
    print("signal {}, noise {}, fec {}, msc {}, over {}, nds {}".format(
        signal_val, noise_val, fec_val, msc_val, over_val, nds_val))
    with open('svm/objective_val.txt', 'w') as file:
        file.write("signal {}, noise {}, fec {}, msc {}, over {}, nds {}".format(
            signal_val, noise_val, fec_val, msc_val, over_val, nds_val))

    plot_res(y_test, y_pred)
    
