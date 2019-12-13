# Load libraries

import pydotplus
from IPython.display import Image
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation
from sklearn.externals.six import StringIO
from sklearn.tree import DecisionTreeClassifier, export_graphviz



criterion = "entropy"
# criterion = "gini"
min_samples_split = 30
max_depth = 3
label_species = ['AdenomeraAndre', 'AdenomeraHylaedactylus', 'Ameeregatrivittata', 'HylaMinuta', 'HypsiboasCinerascens',
                 'HypsiboasCordobae',
                 'LeptodactylusFuscus', 'OsteocephalusOophagus', 'Rhinellagranulosa', 'ScinaxRuber']

# Decision Tree
def decision_tree_new(x_train, x_test, y_train, y_test, feature_names,pos):
    # Split dataset into training set and test set
    # Create Decision Tree classifer object
    # clf = DecisionTreeClassifier()
    clfs = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)

    print("------------------------Decision Tree--------------------------------")

    # Train Decision Tree Classifer
    clf_train = clfs.fit(x_train, y_train)

    # Predict the response for test dataset
    y_pred = clf_train.predict(x_test)

    print("Predicted test data: ", y_pred)

    # predicting a new value

    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:  >>", metrics.accuracy_score(y_test, y_pred) * 100)

    # The score method returns the accuracy of the model
    # score = clfs.score(x_test, y_test)
    #
    # print("Score  >>", score)

    dot_data = StringIO()

    export_graphviz(clfs, out_file=dot_data,
                    filled=True, rounded=True, node_ids='True', proportion='True',
                    special_characters=True, class_names=label_species,
                    feature_names=label_species)
    # special_characters=True, class_names=map(str, recordNum),

    # data show
    graphs = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graphs.write_png(str(pos)+'decision_out.png')
    Image(graphs.create_png())




    print("------------------------Decision Tree End--------------------------------")
