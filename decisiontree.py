# Load libraries

import pydotplus
from IPython.display import Image
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation
from sklearn.externals.six import StringIO
from sklearn.tree import DecisionTreeClassifier, export_graphviz



criterion = "entropy"
#criterion = "gini"
min_samples_split = 30
max_depth = 3

# Decision Tree
def decision_tree_new(x_train, x_test, y_train, y_test, feature_names,pos):
    # Split dataset into training set and test set
    # Create Decision Tree classifer object
    # clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)
    clfs = DecisionTreeClassifier()
    clfg = DecisionTreeClassifier()
    clff = DecisionTreeClassifier()

    # print("------------------------Decision Tree--------------------------------")

    # Train Decision Tree Classifer
    cls_train = clfs.fit(x_train[2], y_train[2])
    clg_train = clfg.fit(x_train[1], y_train[1])
    clf_train = clff.fit(x_train[0], y_train[0])

    # predicting a new value

    ys_pred = cls_train.predict(x_test[2])
    yg_pred = clg_train.predict(x_test[1])
    yf_pred = clf_train.predict(x_test[0])

    # print("Pre:  spec >>", ys_pred)
    # print("Pre:  gen >>", yg_pred)
    # print("Pre:  fam >>", yf_pred)

    Y_pred = [ys_pred, yg_pred, yf_pred]

    i = 0
    # Model Accuracy, how often is the classifier correct?
    while i < 3:
        print("DecisionTree Accuracy: >>", metrics.accuracy_score(y_test[i], Y_pred[2-i]))
        i += 1

    # The score method returns the accuracy of the model
    # score = clfs.score(x_test, y_test)
    #
    # print("Score  >>", score)

    dot_datas = StringIO()
    dot_datag = StringIO()
    dot_dataf = StringIO()

    export_graphviz(clfs, out_file=dot_datas,
                    filled=True, rounded=True, node_ids='True', proportion='True',
                    special_characters=True,
                    feature_names=feature_names)
    export_graphviz(clfg, out_file=dot_datag,
                    filled=True, rounded=True, node_ids='True', proportion='True',
                    special_characters=True,
                    feature_names=feature_names)
    export_graphviz(clff, out_file=dot_dataf,
                    filled=True, rounded=True, node_ids='True', proportion='True',
                    special_characters=True,
                    feature_names=feature_names)
    # special_characters=True, class_names=map(str, recordNum),

    # data show
    # graphs = pydotplus.graph_from_dot_data(dot_datas.getvalue())
    # graphs.write_png(str(pos)+'decision_spe_out.png')
    # Image(graphs.create_png())
    # graphf = pydotplus.graph_from_dot_data(dot_datag.getvalue())
    # graphf.write_png(str(pos)+'decision_gen_out.png')
    # Image(graphf.create_png())
    # graphg = pydotplus.graph_from_dot_data(dot_dataf.getvalue())
    # graphg.write_png(str(pos)+'decision_fam_out.png')
    # Image(graphg.create_png())
    # print("------------------------Decision Tree End--------------------------------")

    return Y_pred

