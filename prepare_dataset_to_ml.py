import json
import numpy as np
from frame import Frame
import csv

# varables to set
# data_read - json file with raw samples
# data_write - .csv file with signal features
DATA_READ = "test_stream.json"
DATA_WRITE = "test_stream.csv"

def load_data(data_path):

    with open(data_path) as fp:
        data = json.load(fp)
        
    X = np.array(data["samples"])
    y = np.array(data["labels"])
    mapping = np.array(data["mapping"])

    return X,y,mapping

def save_to_ml_json(write_path):

    # dictionary to store

    data = {
        "zcr": [],
        "ste": [],
        "classification": []

    }

    # load data
    X,y,mapping = load_data(DATA_READ)

    with open(DATA_WRITE, 'w', newline='') as f:

        fieldnames = ['ste', 'zcr', 'classification']
        writer = csv.DictWriter(f, fieldnames = fieldnames)

        writer.writeheader()
        for (samples, c) in zip(X,y):

            # create object
            f = Frame(samples, c)
            f.calculate_frame_parameters()

            # write to csv
            writer.writerow({'ste' : f.ste, 'zcr' : f.zcr, 'classification' : c})

            print("process...")

    


    


# MAIN
if __name__ == "__main__":

    save_to_ml_json(DATA_WRITE)
    print("end")
        
    
