import json
import numpy as np
np.set_printoptions(threshold=np.nan)

from pymatgen.core.structure import IStructure
from pymatgen.core.periodic_table import Element


class StructureToMatrixEncoder():
    def __init__(self, filename, grid_size=1):
        structure = IStructure.from_file(filename)
        self.grid_size = grid_size

        self.sites = structure.sites
        self.all_species = [site.specie.number for site in self.sites]
        self.all_coords = [site.coords for site in self.sites]

        lattice = structure.lattice
        a, b, c = lattice.a, lattice.b, lattice.c
        self.grid_a = self.get_grid_length(a, self.grid_size)
        self.grid_b = self.get_grid_length(b, self.grid_size)
        self.grid_c = self.get_grid_length(c, self.grid_size)

        # total_atom_info = get_cgcnn_atom_info()
        total_atom_info = atomic_info()
        empty_info = np.full(total_atom_info[0].shape, -1)
        empty_info = np.array([empty_info])
        self.total_atom_info = np.concatenate((empty_info, total_atom_info), axis=0)

    def get_grid_length(self, cell_length, grid_size):
        if cell_length % grid_size == 0:
            return int(cell_length / grid_size)
        else:
            return int(cell_length / grid_size) + 1

    def amI_in(self, coord_c, coord_b, coord_a, coords):
        grid_size = self.grid_size
        if coord_a * grid_size <= coords[0] < coord_a * grid_size + grid_size:
            if coord_b * grid_size <= coords[1] < coord_b * grid_size + grid_size:
                if coord_c * grid_size <= coords[2] < coord_c * grid_size + grid_size:
                    return True
        return False

    def show_matrix_structure(self):
        """

        :return: no return, show matrix with atom number at the grid
        """
        structure_matrix = np.zeros(shape=[self.grid_a, self.grid_b, self.grid_c])
        for i, coords in enumerate(self.all_coords):
            for coord_c in range(self.grid_c):
                for coord_b in range(self.grid_b):
                    for coord_a in range(self.grid_a):
                        chk_in = self.amI_in(coord_c, coord_b, coord_a, coords)
                        if chk_in:
                            structure_matrix[coord_a][coord_b][coord_c] = self.all_species[i]
        m_shape = structure_matrix.shape
        for i in range(3):
            structure_matrix = np.delete(structure_matrix, [m_shape[i] - 1], axis=i)
        # print(structure_matrix)
        # print(structure_matrix.shape)

        return structure_matrix

    def get_structure_matrix(self):
        atom_info_len = len(self.total_atom_info[0])
        structure_matrix = np.zeros(shape=[self.grid_a, self.grid_b, self.grid_c, atom_info_len])
        for i, coords in enumerate(self.all_coords):
            for coord_c in range(self.grid_c):
                for coord_b in range(self.grid_b):
                    for coord_a in range(self.grid_a):
                        chk_in = self.amI_in(coord_c, coord_b, coord_a, coords)
                        if chk_in:
                            structure_matrix[coord_a][coord_b][coord_c] = self.total_atom_info[self.all_species[i]]
        m_shape = structure_matrix.shape
        for i in range(3):
            structure_matrix = np.delete(structure_matrix, [m_shape[i] - 1], axis=i)
        # flat_len = (self.grid_a - 1) * (self.grid_b - 1) * (self.grid_c - 1)
        # structure_matrix = np.reshape(structure_matrix, [flat_len, atom_info_len])
        return structure_matrix


def atomic_info():
    """
    Useful attributes: number, full_electroninc_structure, row, group, atomic_mass, atomic_radius,
                       van_der_waals_radius, average_ionic_radius, X (electronegativity)
    :param atoms:
    :return:
    """
    # -- Parsing periodic_table.json@pymatgen
    jstring = open("./Data/periodic_table.json", "r").read()
    js = json.loads(jstring)
    data = {}
    keys = []
    for key in js.keys():
        atomic_no = js[key]['Atomic no']
        keys.append(atomic_no)
        data[js[key]['Atomic no']] = key
    keys.sort()
    atoms = [data[key] for key in keys]

    # -- Electro negativity
    elec_negativities = []
    for atom in atoms:
        if atom in ["He", "Ne", "Ar"]:             # when X == None
            X = 0.
        else:
            X = Element(atom).X
        elec_negativities.append(X)
    onehot_elec_negativities = [numerical_onehot_encoder(0, 4, 9, X) for X in elec_negativities]
    onehot_elec_negativities = np.array(onehot_elec_negativities)

    # -- Atomic radius
    atomic_radius = []
    for atom in atoms:
        if not Element(atom).atomic_radius:         # when atomic_radius == None
            atomic_radius.append(0.)
        else:
            atomic_radius.append(Element(atom).atomic_radius)
    onehot_atomic_radius = [numerical_onehot_encoder(0, 2.6, 9, r) for r in atomic_radius]
    onehot_atomic_radius = np.array(onehot_atomic_radius)

    # -- Row
    atomic_row = [Element(atom).row for atom in atoms]
    onehot_atomic_row = [numerical_onehot_encoder(1, 9, 9, r) for r in atomic_row]
    onehot_atomic_row = np.array(onehot_atomic_row)

    # -- Group
    atomic_group = [Element(atom).group for atom in atoms]
    onehot_atomic_group = [numerical_onehot_encoder(1, 18, 18, g) for g in atomic_group]
    onehot_atomic_group = np.array(onehot_atomic_group)


    # -- Concat all properties
    properties = (onehot_elec_negativities, onehot_atomic_radius, onehot_atomic_row, onehot_atomic_group)
    total_info = np.concatenate(properties, axis=1)

    # shape = (103, num_total_onehot_length)
    # ex.     (103, 47)
    return total_info


def get_cgcnn_atom_info():
    """

    :return: numpy array of atomic information
             each row (index) indicates corresponding to atomic number - 1
    """
    # -- atom_info load from CGCNN
    jstring = open("./Data/atom_init.json", "r").read()
    loaded_dict = json.loads(jstring)
    atom_info = []
    for key in loaded_dict.keys():
        atom_info.append(loaded_dict[key])
    atom_info = np.array(atom_info)
    # -- end
    return atom_info


def numerical_onehot_encoder(min_val, max_val, length, val):
    """

    :param min_val: minimum value of one hot tray
    :param max_val: maximum value of one hot tray
    :param length: size of one hot tray
    :param val: input value
    :return: one hot encoded numpy array
    """
    tray = np.zeros(shape=(length))
    step = (float(max_val) - float(min_val)) / float(length)
    criteria = np.linspace(min_val, max_val, length)

    for i, c in enumerate(criteria):
        if val <= c:
            tray[i] = 1
            break

    return tray


if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("./Data/metal-alloy-db.v1/00Total_DB.csv")
    # df = df.sample(n=len(df))
    df = df.sample(n=10)
    formation_energy = np.array(df['FormationEnergy'].tolist())
    cifs = "./Data/metal-alloy-db.v1/" + df['DBname'] + ".cif"
    cifs = cifs.tolist()
    encoder = StructureToMatrixEncoder(cifs[0], grid_size=1)
    encoder.show_matrix_structure()
    m = encoder.get_structure_matrix()
    print(m.shape)
