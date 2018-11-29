import json
import numpy as np
np.set_printoptions(threshold=np.nan)

from pymatgen.core.structure import IStructure



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

        total_atom_info = get_cgcnn_atom_info()
        empty_info = np.zeros(shape=total_atom_info[0].shape)
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
        print(structure_matrix)
        print(structure_matrix.shape)

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
