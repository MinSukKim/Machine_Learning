# Random Forest library
from sklearn.ensemble import RandomForestClassifier

# metrics: accuracy calculation
from sklearn import metrics
from sklearn.metrics import confusion_matrix

import numpy as np
import pandas as pd

import seaborn as sns

# Random Forest
def random_Forest(x_train, x_test, y_train, y_test, feature_names):
    # n_estimators = the number of trees
    # n_jobs = the number of core
    rfts = RandomForestClassifier(n_estimators=10, random_state=0)
    rftg = RandomForestClassifier(n_estimators=10, random_state=0)
    rftf = RandomForestClassifier(n_estimators=10, random_state=0)

    rfts = rfts.fit(x_train[2], y_train[2])
    rftg = rftg.fit(x_train[1], y_train[1])
    rftf = rftf.fit(x_train[0], y_train[0])


    # Predict the response for test dataset
    y_preds = rfts.predict(x_test[2])
    y_predg = rftg.predict(x_test[1])
    y_predf = rftf.predict(x_test[0])

    Y_pred=[y_preds, y_predg, y_predf]

    i = 0
    # Model Accuracy, how often is the classifier correct?
    while i < 3:
        print("Random Forest Accuracy: >>", metrics.accuracy_score(y_test[i], Y_pred[2-i]))
        i+=1

    # test = set(Y_pred)
    # cm = pd.DataFrame(confusion_matrix(y_test, Y_pred), columns=test, index=test)

    # cm['species'] = np.array([y_pred[i]] for i in y_pred)

    # sns.pairplot(cm, hue='species')

    # sns.heatmap(cm, annot=True)

    return Y_pred