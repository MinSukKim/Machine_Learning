import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix


def k_nearest_neighbor_new(x_train, x_test, y_train, y_test, feature_names):

    classifierS = KNeighborsClassifier(n_neighbors=5)
    classifierG = KNeighborsClassifier(n_neighbors=5)
    classifierF = KNeighborsClassifier(n_neighbors=5)

    classifierS.fit(x_train[0], y_train[0])
    classifierG.fit(x_train[1], y_train[1])
    classifierF.fit(x_train[2], y_train[2])

    y_predf = classifierF.predict(x_test[2])
    y_predg = classifierG.predict(x_test[1])
    y_preds = classifierS.predict(x_test[0])

    Y_pred = [y_preds, y_predg, y_predf]
    # print(confusion_matrix(y_test, y_pred))
    # print(classification_report(y_test, y_pred))
    i = 0
    # Model Accuracy, how often is the classifier correct?
    while i < 3:
        print("K Nearest Neighbor Accuracy: >>", metrics.accuracy_score(y_test[i], Y_pred[i]))
        i += 1

    return Y_pred
