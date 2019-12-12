from main import label_species, label_genus, label_families


def inconsistency(species, genus, families):
    return species == label_species[1] \
           or species == label_species[0] \
           and genus == label_genus[0] \
           and families == label_families[3] \
           or species == label_species[6] \
           and genus == label_genus[4] \
           and families == label_families[3] \
           or species == label_species[9] \
           and genus == label_genus[7] \
           and families == label_families[2] \
           or species == label_species[4] \
           and genus == label_genus[3] \
           and families == label_families[2] \
           or species == label_species[5] \
           and genus == label_genus[3] \
           and families == label_families[2] \
           or species == label_species[7] \
           and genus == label_genus[5] \
           and families == label_families[2] \
           or species == label_species[3] \
           and genus == label_genus[2] \
           and families == label_families[2] \
           or species == label_species[8] \
           and genus == label_genus[6] \
           and families == label_families[0] \
           or species == label_species[2] \
           and genus == label_genus[1] \
           and families == label_families[1]
