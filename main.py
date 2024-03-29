import csv
import numpy as np
import random

# Data split function
from sklearn.model_selection import train_test_split as split

# Different libraries for different algorithms
from decisiontree import decision_tree_new, decision_tree_new_hierarchy
from kNearestNeighbor import k_nearest_neighbor_new, k_nearest_neighbor_new_hierarchy
from randomForest import random_Forest, random_Forest_hierarchy
from sklearn.metrics import confusion_matrix

# This labels are species, genus and families.
from relationCheck import no_inconsistency

label_species = ['AdenomeraAndre', 'AdenomeraHylaedactylus', 'Ameeregatrivittata', 'HylaMinuta', 'HypsiboasCinerascens',
                 'HypsiboasCordobae',
                 'LeptodactylusFuscus', 'OsteocephalusOophagus', 'Rhinellagranulosa', 'ScinaxRuber']
label_genus = ['Adenomera', 'Ameerega', 'Dendropsophus', 'Hypsiboas', 'Leptodactylus', 'Osteocephalus', 'Rhinella',
               'Scinax']
label_families = ['Bufonidae', 'Dendrobatidae', 'Hylidae', 'Leptodactylidae']

feature_names = ['MFCCs_ 1', 'MFCCs_ 2', 'MFCCs_ 3', 'MFCCs_ 4', 'MFCCs_ 5', 'MFCCs_ 6', 'MFCCs_ 7', 'MFCCs_ 8',
                 'MFCCs_ 9', 'MFCCs_10', 'MFCCs_11', 'MFCCs_12', 'MFCCs_13', 'MFCCs_14', 'MFCCs_15', 'MFCCs_16',
                 'MFCCs_17', 'MFCCs_18', 'MFCCs_19', 'MFCCs_20', 'MFCCs_21', 'MFCCs_22']


# Data Loading function
def load_data():
    with open('MFCCS/Frogs_MFCCs.csv') as file:
        reader = csv.reader(file, delimiter=',')

        label_s = []  # y data
        label_g = []  # y data
        label_f = []  # y data
        record = []  # x data

        for i, row in enumerate(reader):
            # Skip the first line in the csv file
            if i != 0:
                # Save MFCCs data from 1 to 22
                record.append(row[:-4])

                # Save species which is labels for this classification
                label_s.append(label_species.index(row[-2]))
                # Save genus which is labels for this classification
                label_g.append(label_genus.index(row[-3]))
                # Save families which is labels for this classification
                label_f.append(label_families.index(row[-4]))

    return np.array(record).astype(float), np.array(label_s).astype(str), np.array(label_g).astype(str), np.array(
        label_f).astype(str)


def data_split(x, y, i):
    # split x and y1 data, test size = ratio(default = 0.25), train_size = the remaining data
    # random default state 123
    j = 0
    x_train = []
    x_test = []
    y_train = []
    y_test = []

    while j < 3:
        x_tr, x_te, y_tr, y_te = split(x, y[j], test_size=i, random_state=123)
        j += 1

        x_train.append(x_tr)
        x_test.append(x_te)
        y_train.append(y_tr)
        y_test.append(y_te)

    return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    # Load data from csv file, X is data, Ys are labels
    X, Y1, Y2, Y3 = load_data()

    # [species, genus, families]
    Y = [Y1, Y2, Y3]

    # i = 0
    # for data in X:
    #     if not inconsistency(Y1[i], Y2[i], Y3[i]):
    #         print(i, Y1[i], Y2[i], Y3[i])
    #     i += 1

    # test data 0.2, 0.3, 0.4%
    data_ratio = [0.2, 0.3, 0.4]
    i = len(data_ratio)

    while i > 0:
        print("Ratio: ", data_ratio[i - 1], '-----------------')
        x_train, x_test, y_train, y_test = data_split(X, Y, data_ratio[i - 1])

        result_decision_tree = decision_tree_new(x_train, x_test, y_train, y_test, feature_names, i)
        count = 0
        for j in range(0, len(result_decision_tree[0])):
           if not no_inconsistency(int(result_decision_tree[0][j]), int(result_decision_tree[1][j]),
                                   int(result_decision_tree[2][j])):
     #          print(j, result_decision_tree[0][j], result_decision_tree[1][j], result_decision_tree[2][j])
     #          print("correct would be ", j, y_train[0][j], y_train[1][j], y_train[2][j])
               count += 1
        print("number of inconsistencies:", count, "in ", len(result_decision_tree[0]), "predictions")

        result_decision_tree_hierarchy = decision_tree_new_hierarchy(x_train, x_test, y_train, y_test, feature_names, i)
        count = 0
        for j in range(0, len(result_decision_tree_hierarchy[0])):
           if not no_inconsistency(int(result_decision_tree_hierarchy[0][j]), int(result_decision_tree_hierarchy[1][j]),
                                   int(result_decision_tree_hierarchy[2][j])):
               print(j, result_decision_tree_hierarchy[0][j], result_decision_tree_hierarchy[1][j], result_decision_tree_hierarchy[2][j])
               print("correct would be ", j, y_train[0][j], y_train[1][j], y_train[2][j])
               count += 1
        print("number of inconsistencies:", count, "in ", len(result_decision_tree_hierarchy[0]), "predictions")
        count = 0
        result = random_Forest(x_train, x_test, y_train, y_test, feature_names)
        for j in range(0, len(result[0])):
            if not no_inconsistency(int(result[0][j]), int(result[1][j]), int(result[2][j])):

             #  print(j, result[1][j], result[1][j], result[2][j])
             #  print("correct would be ", j, y_train[0][j], y_train[1][j], y_train[2][j])
             count += 1
        print("number of inconsistencies:", count, "in ", len(result[0]), "predictions")
        count = 0

        result_hierarchy = random_Forest_hierarchy(x_train, x_test, y_train, y_test, feature_names)

        for j in range(0, len(result_hierarchy[0])):
           if not no_inconsistency(int(result_hierarchy[0][j]), int(result_hierarchy[1][j]), int(result_hierarchy[2][j])):
               print(j, result_hierarchy[1][j], result_hierarchy[1][j], result_hierarchy[2][j])
               print("correct would be ", j, y_train[0][j], y_train[1][j], y_train[2][j])
               count += 1
        print("number of inconsistencies:", count, "in ", len(result_hierarchy[0]), "predictions")
        count = 0
        result_k_nearest_neighbor = k_nearest_neighbor_new(x_train, x_test, y_train, y_test, feature_names)
        for j in range(0, len(result_k_nearest_neighbor[0])):
          if not no_inconsistency(int(result_k_nearest_neighbor[0][j]),
                                  int(result_k_nearest_neighbor[1][j]),
                                  int(result_k_nearest_neighbor[2][j])):
             #print(j, result_k_nearest_neighbor[0][j],
             #      result_k_nearest_neighbor[1][j],
             #      result_k_nearest_neighbor[2][j])
             #print("correct would be ", j, y_train[0][j], y_train[1][j], y_train[2][j])
              count += 1
        print("number of inconsistencies:", count, "in ", len(result_k_nearest_neighbor[0]), "predictions")
        count = 0

        result_k_nearest_neighbor_hierarchy = k_nearest_neighbor_new_hierarchy(x_train, x_test, y_train, y_test, feature_names)

        for j in range(0, len(result_k_nearest_neighbor_hierarchy[0])):
            if not no_inconsistency(int(result_k_nearest_neighbor_hierarchy[0][j]),
                                    int(result_k_nearest_neighbor_hierarchy[1][j]),
                                    int(result_k_nearest_neighbor_hierarchy[2][j])):
                print(j, result_k_nearest_neighbor_hierarchy[0][j],
                      result_k_nearest_neighbor_hierarchy[1][j],
                      result_k_nearest_neighbor_hierarchy[2][j])
                print("correct would be ", j, y_train[0][j], y_train[1][j], y_train[2][j])
                count += 1
        print("number of inconsistencies:", count, "in ", len(result_k_nearest_neighbor_hierarchy[0]), "predictions")
        count = 0
        i -= 1
