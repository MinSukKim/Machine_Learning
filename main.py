import csv
import numpy as np
import random

# Data split function
from sklearn.model_selection import train_test_split as split

# Different libraries for different algorithms
from decisiontree import decision_tree_new
from kNearestNeighbor import k_nearest_neighbor_new
from randomForest import random_Forest

# This labels are species, genus and families.
from relationCheck import inconsistency

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
    Y = [Y1, Y2, Y3]
    i = 0

    for data in X:
        if not inconsistency(Y1[i], Y2[i], Y3[i]):
            print(i, Y1[i], Y2[i], Y3[i])
        i += 1

    # test data 0.2, 0.3, 0.4%
    data_ratio = [0.2, 0.3, 0.4]
    i = len(data_ratio)

    while i > 0:
        print("Ratio: ", data_ratio[i - 1])
        x_train, x_test, y_train, y_test = data_split(X, Y, data_ratio[i - 1])

        decisiontree_results = decision_tree_new(x_train, x_test, y_train, y_test, feature_names, i)
        print("result decisiontree_results: ", decisiontree_results)

        for dt in range(0, len(decisiontree_results)):
            if not inconsistency(decisiontree_results[0][dt], decisiontree_results[1][dt], decisiontree_results[2][dt]):
                print(dt, decisiontree_results[0][dt], decisiontree_results[1][dt], decisiontree_results[2][dt])
            dt += 1


        # result = random_Forest(x_train, x_test, y_train, y_test, feature_names)
        # print("result : ", result)
        #
        # for i in range(0, len(result)):
        #     if not inconsistency(result[0][i], result[1][i], result[2][i]):
        #         print(i, result[0][i], result[1][i], result[2][i])
        #     i += 1


        # k_nearest_neighbor_new(x_train, x_test, y_train, y_test, feature_names)

        i -= 1
