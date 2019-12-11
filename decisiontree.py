# Load libraries
import csv
import csv
from random import randrange
import math
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import numpy as np
import pydotplus
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

# This labels are species, genus and families.
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_graphviz

feature_names = ['MFCCs_ 1', 'MFCCs_ 2', 'MFCCs_ 3', 'MFCCs_ 4', 'MFCCs_ 5', 'MFCCs_ 6', 'MFCCs_ 7', 'MFCCs_ 8', 'MFCCs_ 9', 'MFCCs_10', 'MFCCs_11', 'MFCCs_12', 'MFCCs_13', 'MFCCs_14', 'MFCCs_15', 'MFCCs_16', 'MFCCs_17', 'MFCCs_18', 'MFCCs_19', 'MFCCs_20', 'MFCCs_21', 'MFCCs_22']





# Decision Tree
def decision_tree_new( labelmain,X,y1, y2, y3, p):
    # Split dataset into training set and test set
    XS_train, XS_test, yS_train, yS_test = train_test_split(X, y1, test_size=0.1,
                                                        random_state=1)  # 70% training and 30% test
    Xg_train, Xg_test, yg_train, yg_test = train_test_split(X, y2, test_size=0.2,
                                                            random_state=1)  # 70% training and 30% tes
    Xf_train, Xf_test, yf_train, yf_test = train_test_split(X, y3, test_size=0.3,
                                                            random_state=1)  # 70% training and 30% tes
    # Create Decision Tree classifer object
    #clf = DecisionTreeClassifier()
    # Create Decision Tree classifer object
    clfs = DecisionTreeClassifier(criterion="entropy", max_depth=3)
    clfg = DecisionTreeClassifier(criterion="entropy", max_depth=3)
    clff = DecisionTreeClassifier(criterion="entropy", max_depth=3)

    # Train Decision Tree Classifer
    clfs = clfs.fit(XS_train,yS_train)
    clfg = clfg.fit(Xg_train,yg_train)
    clff = clff.fit(Xf_train,yf_train)

    #Predict the response for test dataset
    ys_pred = clfs.predict(XS_test)
    yg_pred = clfg.predict(Xg_test)
    yf_pred = clff.predict(Xf_test)


    # Model Accuracy, how often is the classifier correct?
    print("Accuracy: S >>", metrics.accuracy_score(yS_test, ys_pred))
    print("Accuracy: g >>", metrics.accuracy_score(yg_test, yg_pred))
    print("Accuracy: f >>", metrics.accuracy_score(yf_test, yf_pred))


    dot_datas = StringIO()
    dot_datag = StringIO()
    dot_dataf = StringIO()

    export_graphviz(clfs, out_file=dot_datas,
                    filled=True, rounded=True,
                    special_characters=True,
                    feature_names= feature_names)
    export_graphviz(clfg, out_file=dot_datag,
                    filled=True, rounded=True,
                    special_characters=True,
                    feature_names= feature_names)
    export_graphviz(clff, out_file=dot_dataf,
                    filled=True, rounded=True,
                    special_characters=True,
                    feature_names= feature_names)

    graphs = pydotplus.graph_from_dot_data(dot_datas.getvalue())
    graphs.write_png('s.png')
    Image(graphs.create_png())
    graphg = pydotplus.graph_from_dot_data(dot_datag.getvalue())
    graphg.write_png('g.png')
    Image(graphs.create_png())
    graphf = pydotplus.graph_from_dot_data(dot_dataf.getvalue())
    graphf.write_png('f.png')
    Image(graphs.create_png())



