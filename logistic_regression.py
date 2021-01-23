from sklearn.linear_model import LogisticRegression
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

# params to plots
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

# data
DATA_TRAIN_FEATURES = 'train_features.csv'
DATA_TEST_FEATURES = 'eval_features.csv'
DATA_STREAM_FEAUTERS = 'test_stream.csv'


def load_data_features(data_path):

    dataset = pd.read_csv(data_path)

    X = dataset.iloc[:,:-1].values
    y = dataset.iloc[:,-1].values

    return X,y


def plot_res(y, y_pred):
    
    y = y[1080:1180]
    y_pred = y_pred[1080:1180]
    
    plt.style.use('ggplot')
    fig, ax = plt.subplots()
    ax.set_title('wykrywanie aktywności głosowej metodą regresji logistycznej na wycinku sygnału odebranego')
    ax.set_xlabel('numer ramki')
    ax.set_ylabel('mowa/brak mowy')
    ax.plot(y_pred,color = 'b', label = 'stan faktyczny')
    ax.plot(y_pred,color = 'r', linestyle = '--', label = 'decyzja algorytmu')
    ax.set_yticks([0,1])
    ax.set_yticklabels(['brak mowy', 'mowa'])
    ax.legend(loc ="lower right")
    fig.savefig('plots_final/logistic.png')
    plt.show()

# analyze signal
def classify_by_feature(X_train, X_test, y_train, y_test):
    
    # scale the features

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # classifier and fitting
    classifier = LogisticRegression(random_state = 0)
    classifier.fit(X_train, y_train)

    # predictions
    y_pred = classifier.predict(X_test)

    # confusion matrix

    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)

    plot_res(y_test, y_pred)

    signal, noise, fec, msc, over, nds = analyze_objective_params(y_test, y_pred)
    print("signal {}, noise {}, fec {}, msc {}, over {}, nds {}".format(
            signal, noise, fec, msc, over, nds))

    return classifier, sc

# plot dataset
def visualize_dataset(x, y):
    
    X_set, y_set = x, y
    
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], s = 1,
                   c = ListedColormap(('red', 'green'))(i), label = j)

    plt.title('Rozkład parametrów STE i ZCR danych treningowych')
    plt.xlabel('STE')
    plt.ylabel('ZCR')
    plt.legend()
    plt.show()




if __name__ == "__main__":

    # load train dataset
    X_train_features, y_train_features = load_data_features(DATA_TRAIN_FEATURES)

    # load test data
    X_test_features, y_test_features = load_data_features(DATA_TEST_FEATURES)

    X_stream, y_stream = load_data_features(DATA_STREAM_FEAUTERS)

    # visualize dataset
    visualize_dataset(X_train_features, y_train_features)

    # analyze validation data
    classifier, scaler = classify_by_feature(X_train_features, X_test_features, y_train_features, y_test_features)

    # analyze test data
    classifier1, scaler1 = classify_by_feature(X_train_features, X_stream, y_train_features, y_stream)
    




    




  
