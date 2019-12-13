import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix


def k_nearest_neighbor_new(x_train, x_test, y_train, y_test, feature_names):
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print("K Nearest Neighbor Accuracy: >>", metrics.accuracy_score(y_test, y_pred))
