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
from sklearn.model_selection import GridSearchCV
from objective_params import analyze_objective_params

DATA_TRAIN_SAMPLES = 'train.json'
DATA_TEST_SAMPLES = 'eval_rand.json'
DATA_STREAM = 'test_stream.json'


def load_data_samples(data_path):

    with open(data_path) as fp:
        data = json.load(fp)

    X = np.array(data["samples"])
    y = np.array(data["labels"])
    mapping = np.array(data["mapping"])

    return X, y, mapping


if __name__ == "__main__":

    # Load data
    X_train, y_train, map_test = load_data_samples(DATA_TRAIN_SAMPLES)
    X_test, y_test, map_test = load_data_samples(DATA_TEST_SAMPLES)
    X_stream, y_stream, map_stream = load_data_samples(DATA_STREAM)

    # scale the features
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    X_stream = sc.transform(X_stream)

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=10, stop=100, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [2, 4]
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    # Create the param grid
    param_grid = {'n_estimators': n_estimators,
                  'max_features': max_features,
                  'max_depth': max_depth,
                  'min_samples_split': min_samples_split,
                  'min_samples_leaf': min_samples_leaf,
                  'bootstrap': bootstrap}

    rf_Model = RandomForestClassifier()

    rf_Grid = GridSearchCV(
        estimator=rf_Model, param_grid=param_grid, cv=3, verbose=2, n_jobs=4)

    rf_Grid.fit(X_train, y_train)

    rf_Grid.best_params_

    print(f'Train Accuracy - : {rf_Grid.score(X_train,y_train):.3f}')
    print(f'Test Accuracy - : {rf_Grid.score(X_test,y_test):.3f}')
    print(f'Best params - :{rf_Grid.best_params_}')
    

    best_grid = RandomForestClassifier(n_estimators= rf_Grid.best_params_['n_estimators'],
                                       max_features= rf_Grid.best_params_['max_features'],
                                       max_depth= rf_Grid.best_params_['max_depth'],
                                       min_samples_split= rf_Grid.best_params_['min_samples_split'],
                                       min_samples_leaf= rf_Grid.best_params_['min_samples_leaf'],
                                       bootstrap= rf_Grid.best_params_['bootstrap'])

    print("confussion matrix")
    y_pred = best_grid.predict(X_test, y_test)
    cm = confusion_matrix(y_test, y_pred)

    print("calculate test stream")
    y_pred_stream = best_grid.predict(X_stream, y_stream)
    cm = confusion_matrix(y_stream, y_pred_stream)
    with open('random_forest/conf_test.txt', 'w') as file:
        file.write("True Neg {}, False Pos {}, False neg {}, True Pos {}".format(cm[0][0], cm[0][1], cm[1][0], cm[1][1]))

    signal, noise, fec, msc, over, nds = analyze_objective_params(y_stream,y_pred_stream)

    print("signal {}, noise {}, fec {}, msc {}, over {}, nds {}".format(signal, noise, fec, msc, over, nds))
    with open('random_forest/objective.txt', 'w') as file:
        file.write("signal {}, noise {}, fec {}, msc {}, over {}, nds {}".format(signal, noise, fec, msc, over, nds))


