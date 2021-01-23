import numpy as np
import matplotlib.pyplot as plt
import math
import json
from frame import Frame
from sklearn.metrics import confusion_matrix, accuracy_score
from objective_params import analyze_objective_params
import matplotlib.pylab as pylab

# paths
DATA_PATH_TEST = r"test_stream.json"
CALIBRATE_PATH = r"./calibration.json"
DATA_PATH = r"eval_rand.json"

# params to plot
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

# load data
def load_data(data_path):

    with open(data_path) as fp:
        data = json.load(fp)

    X = np.array(data["samples"])
    y = np.array(data["labels"])
    mapping = np.array(data["mapping"])

    print("loaded data: legth: {}".format(len(X)))

    return X,y,mapping

# calibrate treshold
def calibrate_param(data_path):

    # load data
    X,y,mapping = load_data(data_path)

    full_band_energy = 0
    unvoiced_buffer = []
    
    for i in range (N):
        f = Frame(X[i], y[i])
        temp = f.calculate_full_band_energy(X[i])

        # add noise frame to noise buffer
        unvoiced_buffer.append(temp)

        full_band_energy += temp

    full_band_avarage = full_band_energy / N

    return full_band_avarage, unvoiced_buffer

# calculate p value
def p_actualize(var_new, var_old):
    dec = var_new / var_old
    if dec >= 1.25:
        return 0.25
    elif dec < 1.25 and dec >= 1.1:
        return 0.2
    elif dec < 1.1 and dec >= 1:
        return 0.15
    else:
        return 0.10

# new treshold
def calc_new_threshold(e_old, e_silence, p):
    return (1-p)*e_old + p*e_silence

# plot VAD's decision
def plot_res(y, y_pred):
    
    y = y[1080:1180]
    y_pred = y_pred[1080:1180]
    
    plt.style.use('ggplot')
    fig, ax = plt.subplots()
    ax.set_title('wykrywanie aktywności głosowej metodą pomiaru adaptacyjnego pomiaru energii')
    ax.set_xlabel('numer ramki')
    ax.set_ylabel('mowa/brak mowy')
    ax.plot(y,color = 'b', label = 'stan faktyczny')
    ax.plot(y_pred,color = 'r', linestyle = '--', label = 'decyzja algorytmu')
    ax.set_yticks([0,1])
    ax.set_yticklabels(['brak mowy', 'mowa'])
    ax.legend(loc ="lower right")
    fig.savefig('plots_final/aled.png')
    plt.show()

def analyze_signal(data_path):


    # ! new conf matrix from sklearn
    y_pred = []

    # Set thresholds
    p = 0.1
    e_silence = 0
    
    e_r_old, unvoiced_buffer = calibrate_param(CALIBRATE_PATH)
    var_old = np.var(unvoiced_buffer)

    # set init treshold
    Frame.full_band_treshold = e_r_old

    X,y,mapping = load_data(data_path)

    # array with classified frames
    frames_processed = []

    # ! Delete it, it's global
    # unvoiced_buffer = []
    vector_threshold = []

    buffor_index = N - 1

    #frame by frame processing
    for (samples, c) in zip(X,y):
        f = Frame(samples,c)
        f.classify_aled_frame(f.samples)
        vector_threshold.append([Frame.full_band_treshold]*FRAME_SIZE)

        # save frame 
        frames_processed.append(f)
        
        # save the prediction value
        y_pred.append(f.prediction)
    
        # check frame and update threshold
        if f.prediction is 1:
            continue

        # set index at 0, when it is the end of the buffer
        if buffor_index == N:
            buffor_index = 0

        unvoiced_buffer[buffor_index] = f.full_band_energy
    
        var_new = np.var(unvoiced_buffer)
        e_silence = f.full_band_energy
        p = p_actualize(var_new, var_old)
        Frame.full_band_treshold = calc_new_threshold(e_r_old, e_silence,p)
        var_old = var_new
        buffor_index += 1

    cm = confusion_matrix(y, y_pred)
    print(accuracy_score(y, y_pred))

    plot_res(y,y_pred)

    signal, noise, fec, msc, over, nds = analyze_objective_params(y, y_pred)
    print("signal {}, noise {}, fec {}, msc {}, over {}, nds {}".format(
        signal, noise, fec, msc, over, nds))


    return frames_processed, vector_threshold, cm
    

if __name__ == "__main__":

    # analyze validation
    frames_processed, vector_threshold, conf_matrix = analyze_signal(DATA_PATH)
    print(conf_matrix)
    # analyze test
    frames_processed, vector_threshold, conf_matrix = analyze_signal(DATA_PATH_TEST)
    print(conf_matrix)