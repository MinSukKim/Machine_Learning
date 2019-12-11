import csv
import numpy as np

# Classification Library
from sklearn import tree

from sklearn.model_selection import train_test_split as split

# This labels are species, genus and families.
from decisiontree import decision_tree_new

label_species = ['AdenomeraAndre', 'AdenomeraHylaedactylus', 'Ameeregatrivittata', 'HylaMinuta', 'HypsiboasCinerascens',
                 'HypsiboasCordobae',
                 'LeptodactylusFuscus', 'OsteocephalusOophagus', 'Rhinellagranulosa', 'ScinaxRuber']
label_genus = ['Adenomera', 'Ameerega', 'Dendropsophus', 'Hypsiboas', 'Leptodactylus', 'Osteocephalus', 'Rhinella',
               'Scinax']
label_families = ['Bufonidae', 'Dendrobatidae', 'Hylidae', 'Leptodactylidae']


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

    return np.array(record).astype(float), np.array(label_s).astype(int), np.array(label_g).astype(int), np.array(
        label_f).astype(int), record


# Decision Tree
def decision_tree(data, y1, y2, y3, p):
    dt = tree.DecisionTreeClassifier(criterion="entropy")

    # Training decision tree with data and label
    dt.fit(data, y1)

    dt.predict(p)
    t = dt.tree_
    # print(t.impurity)


def k_split(x, y1, y2, y3):
    # split x and y1 data, test size = ratio(default = 0.25), train_size = the remaining data
    # random default
    x_train, x_test, y_train, y_test = split(x, y1, test_size=0.2, random_state=123)
    print(x_train.shape)



if __name__ == "__main__":
    # Load data from csv file, X is data, Ys are labels
    X, Y1, Y2, Y3, record = load_data()
    # print(X)

    # Sample Data checking
    predictData = [[1, 0.152936298, -0.105585903, 0.200721915, 0.317201062, 0.880763853, 0.100944641, -0.150062605,
                    -0.171127632, 0.777776436, 0.188654146, 0.075621723, 0.156435925, 0.082245115, 0.135752042,
                    0.024016645, 0.108351107, 0.077622521, 0.000567802, 0.057683975, 0.118680135, 0.014038446
                    ]]

    #decision_tree(X, Y1, Y2, Y3, predictData)
    k_split(X, Y1, Y2, Y3)
    decision_tree_new(record, X, Y1, Y2, Y3, predictData)

