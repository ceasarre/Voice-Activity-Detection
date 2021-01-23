import json
import numpy as np
import matplotlib.pyplot as plt
import random

DATA_PATH = 'eval.json'
WRITE_PATH = './eval_rand.json'

def load_data(data_path):

    with open(data_path) as fp:
        data = json.load(fp)

    X = np.array(data["samples"])
    y = np.array(data["labels"])
    mapping = np.array(data["mapping"])

    return X,y,mapping

def save_to_json(index, X, y, mapping, json_path):
    
    # dictionary to store

    data = {
        "mapping" : mapping.tolist(),
        "labels" : [],
        "samples" : []
    }

    counter = 0

    for i in index:
        data["samples"].append(X[i].tolist())
        data["labels"].append(y[i].tolist())
        counter += 1

    with open(json_path, 'w') as fp:
        json.dump(data, fp)

    fp.close()

    print("\nprzetworzono: {} plików".format(counter))


    


if __name__=="__main__":
    
    # Read the data
    X_data, y_data, mapping = load_data(DATA_PATH)

    # Read data length
    data_len = len(X_data)

    # rand index to create new JSON file
    index = random.sample(range(data_len), data_len)

    # create new pseudo stream
    save_to_json(index, X_data, y_data, mapping, WRITE_PATH)

    print("c")


    

