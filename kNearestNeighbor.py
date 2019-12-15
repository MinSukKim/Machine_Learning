import numpy as np
from numpy import array
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error


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
        if i==0:
            print("Confusion matrix spec")
        elif i==1:
            print ("Confusion matrix genus")
        else:
            print("Confusion matrix family")
        print(confusion_matrix(Y_pred[i], y_test[i]))
        i += 1

    return Y_pred


def k_nearest_neighbor_new_hierarchy(x_train, x_test, y_train, y_test, feature_names):
    classifierS = KNeighborsClassifier(n_neighbors=5)
    classifierG = KNeighborsClassifier(n_neighbors=5)
    classifierF = KNeighborsClassifier(n_neighbors=5)

    classifierS.fit(x_train[0], y_train[0])
    classifierG.fit(x_train[1], y_train[1])
    classifierF.fit(x_train[2], y_train[2])

    y_predf = classifierF.predict(x_test[2])
    y_predg = classifierG.predict(x_test[1])
    for k in range(0, len(y_predf)):
        if int(y_predf[k]) == 0:
            y_predg[k] = 6
        elif int(y_predf[k]) == 1:
            y_predg[k] = 1
        elif int(y_predf[k]) == 2 and (
                not int(y_predg[k]) == 7 and not int(y_predg[k]) == 3 and not int(y_predg[k]) == 5 and not int(
            y_predg[k]) == 2):
            classifierG2 = KNeighborsClassifier(n_neighbors=5)
            x_train2 = np.copy(x_train)
            y_train2 = np.copy(y_train)
            rows=[]
            x = 0
            for t in range(0, len(y_train[1])):
                if int(y_train[1][t]) == 7 or int(y_train[1][t]) == 3 or int(y_train[1][t]) == 5 or int(
                        y_train[1][t]) == 2:
                    x += 1
                else:
                    rows.append(t)
            x_train2 = np.delete(x_train2, rows, 1)
            y_train2 = np.delete(y_train2, rows, 1)
            classifierG2.fit(x_train2[1], y_train2[1])
            y_predg2 = classifierG2.predict(x_test[1])
            y_predg[k] = y_predg2[k]

        elif int(y_predf[k]) == 3 and (
                not int(y_predg[k]) == 4 and not int(y_predg[k]) == 0):
            classifierG3 = KNeighborsClassifier(n_neighbors=5)
            y_train3 = np.copy(y_train)
            x_train3 = np.copy(x_train)
            x = 0
            rows=[]
            for t in range(0, len(y_train[1])):
                if int(y_train[1][t]) == 4 or int(y_train[1][t]) == 0:
                    x += 1
                else: rows.append(t)
            x_train3 = np.delete(x_train3, rows, 1)
            y_train3 = np.delete(y_train3, rows, 1)
            classifierG3.fit(x_train3[1], y_train3[1])
            y_predg3 = classifierG3.predict(x_test[1])
            y_predg[k] = y_predg3[k]

    y_preds = classifierS.predict(x_test[0])
    for k in range(0, len(y_predg)):
        if int(y_predg[k]) == 6:
            y_preds[k] = 8
        elif int(y_predg[k]) == 1:
            y_preds[k] = 2
        elif int(y_predg[k]) == 4:
            y_preds[k] = 6
        elif int(y_predg[k]) == 7:
            y_preds[k] = 9
        elif int(y_predg[k]) == 5:
            y_preds[k] = 7
        elif int(y_predg[k]) == 2:
            y_preds[k] = 3
        elif int(y_predg[k]) == 0 and (
                not int(y_preds[k]) == 1 and not int(y_preds[k]) == 0):
            classifierS0 = KNeighborsClassifier(n_neighbors=5)
            y_train0 = np.copy(y_train)
            x_train0 = np.copy(x_train)
            x = 0
            rows = []
            for t in range(0, len(y_train[0])):
                if int(y_train[0][t]) == 1 or int(y_train[0][t]) == 0:
                    x += 1
                else: rows.append(t)
            x_train0 = np.delete(x_train0, rows, 1)
            y_train0 = np.delete(y_train0, rows, 1)
            classifierS0.fit(x_train0[0], y_train0[0])
            y_preds0 = classifierS0.predict(x_test[0])
            y_preds[k] = y_preds0[k]

        elif int(y_predg[k]) == 3 and (
                not int(y_preds[k]) == 5 and not int(y_preds[k]) == 4):
            classifierS3 = KNeighborsClassifier(n_neighbors=5)
            y_train3 = np.copy(y_train)
            x_train3 = np.copy(x_train)
            x = 0
            rows = []
            for t in range(0, len(y_train[0])):
                if int(y_train[0][t]) == 5 or int(y_train[0][t]) == 4:
                    x += 1
                else: rows.append(t)
            x_train3 = np.delete(x_train3, rows, 1)
            y_train3 = np.delete(y_train3, rows, 1)
            classifierS3.fit(x_train3[0], y_train3[0])
            y_preds3 = classifierS3.predict(x_test[0])
            y_preds[k] = y_preds3[k]

    Y_pred = [y_preds, y_predg, y_predf]
    # print(confusion_matrix(y_test, y_pred))
    # print(classification_report(y_test, y_pred))
    i = 0
    # Model Accuracy, how often is the classifier correct?
    while i < 3:
        print(
            "K Nearest Neighbor with hierarchical correction Accuracy: >>",
            metrics.accuracy_score(y_test[i], Y_pred[i]))
        if i==0:
            print("Confusion matrix spec")
        elif i==1:
            print ("Confusion matrix genus")
        else:
            print("Confusion matrix family")
        print(confusion_matrix(Y_pred[i], y_test[i]))
        i += 1

    return Y_pred
