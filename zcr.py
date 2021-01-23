from frame import Frame
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, auc
from objective_params import analyze_objective_params

# Set params to plot
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

# Paths to Dataset
DATA_PATH = r"eval_rand.json"
DATA_STREAM = r"test_stream.json"
CALIBRATE_PATH = r"./calibration.json"


# Loading data
def load_data(data_path):

    with open(data_path) as fp:
        data = json.load(fp)

    X = np.array(data["samples"])
    y = np.array(data["labels"])
    mapping = np.array(data["mapping"])

    return X, y, mapping


# calibration
def calibrate_param(data_path):

    # load data
    X, y, mapping = load_data(data_path)

    zcr = 0

    # calculate ste and zcr
    for (samples, c) in zip(X, y):
        f = Frame(samples, c)
        temp = f.zero_crossing_rate()
        zcr += temp

    # avarage results to return
    zcr_avarage = zcr / len(X)

    return zcr_avarage

# ploting the VAD's deciscion
def plot_res(y, y_pred):
    
    y = y[1080:1180]
    y_pred = y_pred[1080:1180]
    
    plt.style.use('ggplot')
    fig, ax = plt.subplots()
    ax.set_title('wykrywanie aktywności głosowej metodą pomiaru przejść sygnału przez zero')
    ax.set_xlabel('numer ramki')
    ax.set_ylabel('mowa/brak mowy')
    ax.plot(y,color = 'b', label = 'stan faktyczny')
    ax.plot(y_pred,color = 'r', linestyle = '--', label = 'decyzja algorytmu')
    ax.set_yticks([0,1])
    ax.set_yticklabels(['brak mowy', 'mowa'])
    ax.legend(loc ="lower right")
    fig.savefig('plots_final/zcr.png')
    plt.show()


# Function to analyze the signal
def analyze_signal(data_path):

    # calibrate zcr treshold
    Frame.zcr_treshold = calibrate_param(CALIBRATE_PATH)

    frames_processed = []
    y_pred = []

    # load data
    X, y, mapping = load_data(data_path)

    for(samples, c) in zip(X, y):
        print("processing")
        f = Frame(samples, c)
        f.classify_zcr()
        frames_processed.append(f)
        y_pred.append(f.prediction)

    cm = confusion_matrix(y, y_pred)
    print(accuracy_score(y, y_pred))
    print(cm)


    plot_res(y, y_pred)
    signal, noise, fec, msc, over, nds = analyze_objective_params(y, y_pred)
    print("signal {}, noise {}, fec {}, msc {}, over {}, nds {}".format(
        signal, noise, fec, msc, over, nds))

    return frames_processed


# MAIN
if __name__ == "__main__":

    # Validation data
    processed = analyze_signal(DATA_PATH)
    # test data
    processed2 = analyze_signal(DATA_STREAM)
