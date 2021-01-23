from scipy.signal.windows import hamming
import numpy as np
import math
import matplotlib.pyplot as plt


class Frame:

    # Thresholds init
    zcr_treshold = 0
    ste_treshold = 0
    full_band_treshold = 0

    def __init__(self, samples, classification):

        self.frame_size = len(samples)
        self.samples = np.array(samples)
        self.classification = classification
        self.prediction = 0
        self.zcr = 0
        self.ste = 0
        self.full_band_energy = 0

        self.hamming_window = hamming(self.frame_size)

        # vectors to plots
        self.classification_vector = []
        self.prediction_vector = []

    # calculate Zero Crossing Rate
    def zero_crossing_rate(self):
        return sum(abs(np.diff(self.samples > 0))) / len(self.samples)

    # Function to calculate Short Time Energy
    def short_time_energy(self):
        temp = 0
        energy = 0
        for i, value in enumerate(self.samples):
            temp = pow(self.samples[i]*self.hamming_window[len(self.hamming_window) - 1 - i],2)
            energy += temp
        return energy / len(self.samples)

    # Function to calulate the parameters of the frame
    def calculate_frame_parameters(self):
        self.zcr = self.zero_crossing_rate()
        self.ste = self.short_time_energy()



    # decode to string voiced / unvoiced
    def decodeFrame(self, i):
        if(i == 0):
            return 'unvoiced'
        else:
            return 'voiced'

     # Function to print the parameters
    def getParameters(self):
        print('zero crossing rate: {}'.format(self.zcr))
        print('short time energy: {}'.format(self.ste))
        print('prediction: {}'.format(self.decodeFrame(self.prediction)))
        print('classification: {}'.format(self.decodeFrame(self.classification)))
        print('length in samples: {}'.format(len(self.samples)))

    # Function to plot the frame
    def plotFrame(self):
        plt.plot(self.samples)
        plt.grid()
        plt.show()

    def calculate_full_band_energy(self, samples):
        
        temp = 0
        energy = 0
        # hamming_window = hamming(len(samples))

        for i, sample in enumerate(samples):
            temp = pow(sample*hamming_window[int(len(hamming_window) - 1 - i)],2)
            # temp = pow(sample,2)
            energy += temp

            return energy / len(samples)

                     


    def classify_aled_frame(self, samples):

    
        energy = self.calculate_full_band_energy(samples)

        if(energy < Frame.full_band_treshold):
             self.prediction = 1
        else:
            self.prediction = 0

        self.full_band_energy = energy


    def classify_zcr(self):
        self.zcr = self.zero_crossing_rate()

        if self.zcr < Frame.zcr_treshold:
            self.prediction = 1
        else:
            self.prediction = 0

    def classify_ste(self):
        self.ste = self.short_time_energy()

        if self.ste < Frame.ste_treshold:
            self.prediction = 1
        else:
            self.prediction = 0

    # Classify as voiced only if both detectors are 1
    def classify(self):

        self.calculate_frame_parameters()
        if self.zcr < Frame.zcr_treshold and self.ste < Frame.ste_treshold:
            self.prediction = 1

        else:
            self.prediction = 0

                