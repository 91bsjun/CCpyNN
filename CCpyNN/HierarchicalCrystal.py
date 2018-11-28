import os
import json
import numpy as np
np.set_printoptions(threshold=np.nan)

from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import IStructure

structure = IStructure.from_file("./Data/from_cgcnn/1000041.cif")
grid_size = 1

sites = structure.sites
all_species = [site.specie.number for site in sites]
all_coords = [site.coords for site in sites]

lattice = structure.lattice
a, b, c = lattice.a, lattice.b, lattice.c


def get_grid_length(cell_length, grid_size):
    if cell_length % grid_size == 0:
        return int(cell_length / grid_size)
    else:
        return int(cell_length / grid_size) + 1


grid_a, grid_b, grid_c = get_grid_length(a, grid_size), get_grid_length(b, grid_size), get_grid_length(c, grid_size)
structure_matrix = np.zeros(shape=[grid_a, grid_b, grid_c])


def amI_in(grid_size, coord_c, coord_b, coord_a, coords):
    if coord_a * grid_size <= coords[0] < coord_a * grid_size + grid_size:
        if coord_b * grid_size <= coords[1] < coord_b * grid_size + grid_size:
            if coord_c * grid_size <= coords[2] < coord_c * grid_size + grid_size:
                return True
    return False


for i, coords in enumerate(all_coords):
    for coord_c in range(grid_c):
        for coord_b in range(grid_b):
            for coord_a in range(grid_a):
                chk_in = amI_in(grid_size, coord_c, coord_b, coord_a, coords)
                if chk_in:
                    structure_matrix[coord_a][coord_b][coord_c] = all_species[i]

print(structure_matrix.shape)
r = structure_matrix[:-1]
print(r.shape)
r = r[:][:-1]
print(r.shape)
r = r[:][:][:-1]
print(r.shape)


