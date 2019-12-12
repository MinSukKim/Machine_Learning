# from sklearn.externals.six import StringIO
# from IPython.display import Image
# import pydotplus

# metrics: accuracy calculation
from sklearn import metrics

# This labels are species, genus and families.
from sklearn.tree import DecisionTreeClassifier, export_graphviz




# Decision Tree
def decision_tree_new(x_train, x_test, y_train, y_test, feature_names):
    clfs = DecisionTreeClassifier(criterion="entropy", max_depth=3)

    # Train Decision Tree Classifer
    clfs = clfs.fit(x_train, y_train)

    # Predict the response for test dataset
    ys_pred = clfs.predict(x_test)
    # print(ys_pred)
    # Model Accuracy, how often is the classifier correct?
    print("Decision Tree Accuracy: >>", metrics.accuracy_score(y_test, ys_pred))

    # dot_datas = StringIO()
    #
    # export_graphviz(clfs, out_file=dot_datas,
    #                 filled=True, rounded=True,
    #                 special_characters=True,
    #                 feature_names=feature_names)
    #
    # graphs = pydotplus.graph_from_dot_data(dot_datas.getvalue())
    # graphs.write_png('decisionTree.png')
    # Image(graphs.create_png())