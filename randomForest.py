# Random Forest library
from sklearn.ensemble import RandomForestClassifier

# metrics: accuracy calculation
from sklearn import metrics

from sklearn import tree

# Random Forest
def random_Forest(x_train, x_test, y_train, y_test, feature_names, X, Y):
    # n_estimators = the number of trees
    # n_jobs = the number of core
    rft = RandomForestClassifier(n_estimators=5, random_state=2)

    rft = rft.fit(x_train, y_train)

    # Predict the response for test dataset
    y_pred = rft.predict(x_test)
    # print(ys_pred)
    # Model Accuracy, how often is the classifier correct?
    print("Random Forest Accuracy: >>", metrics.accuracy_score(y_test, y_pred))

    # tree.plot_tree(rft)

