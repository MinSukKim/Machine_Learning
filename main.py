import csv
import numpy as np

# Classification Library
from sklearn import tree

# Data split function
from sklearn.model_selection import train_test_split as split

# This labels are species, genus and families.
from decisiontree import decision_tree_new

# Random Forest library
from sklearn.ensemble import RandomForestClassifier

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

    return np.array(record).astype(float), np.array(label_s).astype(int), np.array(label_g).astype(int), np.array(
        label_f).astype(int)


def data_split(x, y1, y2, y3):
    # split x and y1 data, test size = ratio(default = 0.25), train_size = the remaining data
    # random default state 123
    x_train, x_test, y_train, y_test = split(x, y1, test_size=0.4, random_state=123)

    return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    # Load data from csv file, X is data, Ys are labels
    X, Y1, Y2, Y3 = load_data()

    x_train, x_test, y_train, y_test = data_split(X, Y1, Y2, Y3)
    decision_tree_new(x_train, x_test, y_train, y_test, feature_names)

