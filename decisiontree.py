# Load libraries
import numpy as np
import pydotplus
from IPython.display import Image
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation
from sklearn.externals.six import StringIO
from sklearn.metrics import confusion_matrix, mean_absolute_error
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
    cls_train = clfs.fit(x_train[0], y_train[0])
    clg_train = clfg.fit(x_train[1], y_train[1])
    clf_train = clff.fit(x_train[2], y_train[2])

    # predicting a new value

    ys_pred = cls_train.predict(x_test[0])
    yg_pred = clg_train.predict(x_test[1])
    yf_pred = clf_train.predict(x_test[2])

    # print("Pre:  spec >>", ys_pred)
    # print("Pre:  gen >>", yg_pred)
    # print("Pre:  fam >>", yf_pred)

    Y_pred = [ys_pred, yg_pred, yf_pred]

    i = 0
    # Model Accuracy, how often is the classifier correct?
    while i < 3:
        print("DecisionTree Accuracy: >>", metrics.accuracy_score(y_test[i], Y_pred[i]))
        if i==0:
            print("Confusion matrix spec")
        elif i==1:
            print ("Confusion matrix genus")
        else:
            print("Confusion matrix family")
        print(confusion_matrix(Y_pred[i], y_test[i]))
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

def decision_tree_new_hierarchy(x_train, x_test, y_train, y_test, feature_names,pos):
    # Split dataset into training set and test set
    # Create Decision Tree classifer object
    # clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)
    classifierS = DecisionTreeClassifier()
    classifierG = DecisionTreeClassifier()
    classifierF = DecisionTreeClassifier()

    # print("------------------------Decision Tree--------------------------------")

    # Train Decision Tree Classifer
    y_preds = classifierS.fit(x_train[0], y_train[0])
    y_predg = classifierG.fit(x_train[1], y_train[1])
    y_predf = classifierF.fit(x_train[2], y_train[2])

    # predicting a new value

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
            classifierG2 = DecisionTreeClassifier()
            x_train2 = np.copy(x_train)
            y_train2 = np.copy(y_train)
            rows = []
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
            classifierG3 = DecisionTreeClassifier()
            y_train3 = np.copy(y_train)
            x_train3 = np.copy(x_train)
            x = 0
            rows = []
            for t in range(0, len(y_train[1])):
                if int(y_train[1][t]) == 4 or int(y_train[1][t]) == 0:
                    x += 1
                else:
                    rows.append(t)
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
            classifierS0 = DecisionTreeClassifier()
            y_train0 = np.copy(y_train)
            x_train0 = np.copy(x_train)
            x = 0
            rows = []
            for t in range(0, len(y_train[0])):
                if int(y_train[0][t]) == 1 or int(y_train[0][t]) == 0:
                    x += 1
                else:
                    rows.append(t)
            x_train0 = np.delete(x_train0, rows, 1)
            y_train0 = np.delete(y_train0, rows, 1)
            classifierS0.fit(x_train0[0], y_train0[0])
            y_preds0 = classifierS0.predict(x_test[0])
            y_preds[k] = y_preds0[k]

        elif int(y_predg[k]) == 3 and (
                not int(y_preds[k]) == 5 and not int(y_preds[k]) == 4):
            classifierS3 = DecisionTreeClassifier()
            y_train3 = np.copy(y_train)
            x_train3 = np.copy(x_train)
            x = 0
            rows = []
            for t in range(0, len(y_train[0])):
                if int(y_train[0][t]) == 5 or int(y_train[0][t]) == 4:
                    x += 1
                else:
                    rows.append(t)
            x_train3 = np.delete(x_train3, rows, 1)
            y_train3 = np.delete(y_train3, rows, 1)
            classifierS3.fit(x_train3[0], y_train3[0])
            y_preds3 = classifierS3.predict(x_test[0])
            y_preds[k] = y_preds3[k]

    Y_pred = [y_preds, y_predg, y_predf]

    i = 0
    # Model Accuracy, how often is the classifier correct?
    while i < 3:
        print("DecisionTree with hierarchical correction Accuracy: >>", metrics.accuracy_score(y_test[i], Y_pred[i]))
        if i==0:
            print("Confusion matrix spec")
        elif i==1:
            print ("Confusion matrix genus")
        else:
            print("Confusion matrix family")
        print(confusion_matrix(Y_pred[i], y_test[i]))
        i += 1

    # The score method returns the accuracy of the model
    # score = clfs.score(x_test, y_test)
    #
    # print("Score  >>", score)

    dot_datas = StringIO()
    dot_datag = StringIO()
    dot_dataf = StringIO()

    export_graphviz(classifierS, out_file=dot_datas,
                    filled=True, rounded=True, node_ids='True', proportion='True',
                    special_characters=True,
                    feature_names=feature_names)
    export_graphviz(classifierG, out_file=dot_datag,
                    filled=True, rounded=True, node_ids='True', proportion='True',
                    special_characters=True,
                    feature_names=feature_names)
    export_graphviz(classifierF, out_file=dot_dataf,
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

