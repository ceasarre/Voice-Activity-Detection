from sklearn.ensemble import RandomForestClassifier
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

# Paths
DATA_TRAIN_SAMPLES = 'train.json'
DATA_TEST_SAMPLES = 'test_stream.json'
DATA_TEST_EVAL = 'eval_rand.json'

# analyze signal, training params found in grid search
def analyze_signal_samples(train_data, test_data):

    X_train, y_train, mapping = load_data_samples(train_data)
    X_test, y_test, mapping = load_data_samples(test_data)

    sc = StandardScaler()
    sc.fit_transform(X_train)
    sc.transform(X_test)

    classifier_samples = RandomForestClassifier(bootstrap=False, max_depth=4,
                                                max_features='auto', min_samples_leaf=2, min_samples_split=2, n_estimators=80)
    classifier_samples.fit(X_train, y_train)

    y_pred = classifier_samples.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)

    return y_pred, y_test

# load data
def load_data_samples(data_path):

    with open(data_path) as fp:
        data = json.load(fp)

    X = np.array(data["samples"])
    y = np.array(data["labels"])
    mapping = np.array(data["mapping"])

    return X, y, mapping

# plot the VAD's decisions
def plot_res(y, y_pred):
    
    y = y[1080:1180]
    y_pred = y_pred[1080:1180]
    
    plt.style.use('ggplot')
    fig, ax = plt.subplots()
    ax.set_title('wykrywanie aktywności głosowej metodą RFC na wycinku sygnału odebranego')
    ax.set_xlabel('numer ramki')
    ax.set_ylabel('mowa/brak mowy')
    ax.plot(y_pred,color = 'b', label = 'stan faktyczny')
    ax.plot(y_pred,color = 'r', linestyle = '--', label = 'decyzja algorytmu')
    ax.set_yticks([0,1])
    ax.set_yticklabels(['brak mowy', 'mowa'])
    ax.legend(loc ="lower right")
    fig.savefig('plots_final/rfc.png')
    plt.show()

if __name__ == "__main__":

    # analyze signal
    y_prediction, y_fact = analyze_signal_samples(
        DATA_TRAIN_SAMPLES, DATA_TEST_SAMPLES)

    # calculate objective params
    signal, noise, fec, msc, over, nds = analyze_objective_params(
        y_fact, y_prediction)

    print("signal {}, noise {}, fec {}, msc {}, over {}, nds {}".format(
        signal, noise, fec, msc, over, nds))
    with open('random_forest/objective.txt', 'w') as file:
        file.write("signal {}, noise {}, fec {}, msc {}, over {}, nds {}".format(
            signal, noise, fec, msc, over, nds))

    # plot vad's decisions
    plot_res(y_fact, y_prediction)
