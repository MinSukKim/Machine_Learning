def no_inconsistency(species, genus, families):
    return species == 1 \
           and genus == 0 \
           and families == 3 \
           or species == 0 \
           and genus == 0 \
           and families == 3 \
           or species == 6 \
           and genus == 4 \
           and families == 3 \
           or species == 9 \
           and genus == 7 \
           and families == 2 \
           or species == 4 \
           and genus == 3 \
           and families == 2 \
           or species == 5 \
           and genus == 3 \
           and families == 2 \
           or species == 7 \
           and genus == 5 \
           and families == 2 \
           or species == 3 \
           and genus == 2 \
           and families == 2 \
           or species == 8 \
           and genus == 6 \
           and families == 0 \
           or species == 2 \
           and genus == 1 \
           and families == 1


def no_inconsistencyGF(genus, families):
    return genus == 0 \
           and families == 3 \
           or genus == 4 \
           and families == 3 \
           or genus == 7 \
           and families == 2 \
           or genus == 3 \
           and families == 2 \
           or genus == 5 \
           and families == 2 \
           or genus == 2 \
           and families == 2 \
           or genus == 6 \
           and families == 0 \
           or genus == 1 \
           and families == 1
