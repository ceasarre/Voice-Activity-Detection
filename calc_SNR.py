from frame import Frame
import json
import numpy as np
from math import log10

# path
DATA_PATH_NOISE = r"unvoice.json"
DATA_PATH_VOICE= r"./voice.json"

# load data
def load_data(data_path):

    with open(data_path) as fp:
        data = json.load(fp)

    X = np.array(data["samples"])
    y = np.array(data["labels"])
    mapping = np.array(data["mapping"])

    return X,y

# main
if __name__ == '__main__':
    X_voice, y_voice = load_data(DATA_PATH_VOICE)
    X_noise, y_noise = load_data(DATA_PATH_NOISE)
    
    energy_voice = 0
    counter = 0
    temp = 0
    for (samples, c) in zip (X_voice,y_voice):
        f = Frame(samples,c)
        temp = f.calculate_full_band_energy(f.samples)
        energy_voice += temp
        counter +=1

    energy_voice = energy_voice/counter
    
    energy_noise = 0
    counter = 0
    temp = 0
    
    for (samples, c) in zip (X_noise,y_noise):
        f = Frame(samples,c)
        temp = f.calculate_full_band_energy(f.samples)
        energy_noise += temp
        counter+=1

    energy_noise = energy_noise/counter
    
    SNR = 20*log10(energy_voice/energy_noise)
    print(SNR)