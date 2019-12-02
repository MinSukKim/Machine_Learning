import csv
import numpy as np

# This labels are species, genus and families.
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
        label_g = []
        label_f = []
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


if __name__ == "__main__":
    # Load data from csv file
    data = load_data()

    # Data checking
    # print(data[1])
