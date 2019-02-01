import os
import json
import numpy as np
np.set_printoptions(threshold=np.nan)
import tensorflow as tf
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import IStructure

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


def structure_encoder(structure, radius, max_neighbor_num):
    all_sites = structure.sites
    all_sites_atom_num = [site.specie.number for site in all_sites]  # atom number of each sites, shape = (sites)
    # increase radious untill len(neighbors) >= max_neighbor_num
    neighbor_satisfaction = False
    while not neighbor_satisfaction:
        all_neighbors = structure.get_all_neighbors(radius, include_index=True)
        for nbr in all_neighbors:
            if len(nbr) < max_neighbor_num:
                radius += 1.
                break
            else:
                neighbor_satisfaction = True

    # parse neighbor information
    neighbor_index = []
    neighbor_distance = []
    neighbor_atom_num = []
    for i, site in enumerate(all_sites):
        neighbors = structure.get_neighbors(site, radius, include_index=True)
        neighbor_index.append([n[2] for n in neighbors[:max_neighbor_num]])
        neighbor_distance.append([n[1] for n in neighbors[:max_neighbor_num]])
        neighbor_atom_num.append([n[0].specie.number for n in neighbors[:max_neighbor_num]])

    neighbor_index = np.array(neighbor_index)
    neighbor_distance = np.array(neighbor_distance)
    neighbor_atom_num = np.array(neighbor_atom_num)


    # -- custom distance info
    # One hot encoding of distance
    onehot_neighbor_distance = []
    for each_site in neighbor_distance:
       tmp = [numerical_onehot_encoder(1, radius, 10, d) for d in each_site]
       onehot_neighbor_distance.append(tmp)
    onehot_neighbor_distance = np.array(onehot_neighbor_distance)  # shape = (sites, neighbors, length)

    # def gaussian_distance(distances):
    #     dmin = 0
    #     dmax = 8
    #     step = 0.2
    #     var = None
    #     assert dmin < dmax
    #     assert dmax - dmin > step
    #     filter = np.arange(dmin, dmax + step, step)
    #     if var is None:
    #         var = step
    #     var = var
    #
    #     return np.exp(-(distances[..., np.newaxis] - filter) ** 2 /
    #                   var ** 2)
    # onehot_neighbor_distance = gaussian_distance(neighbor_distance)

    # -- custom atom_info
    atom_info = atomic_info()  # atomic information of all atoms as order of atomic number

    # -- atom_info load from CGCNN
    # jstring = open("./Data/atom_init.json", "r").read()
    # loaded_dict = json.loads(jstring)
    # atom_info = []
    # for key in loaded_dict.keys():
    #     atom_info.append(loaded_dict[key])
    # atom_info = np.array(atom_info)
    # -- end

    all_sites_atom_info = [atom_info[num - 1] for num in all_sites_atom_num]
    all_sites_atom_info = np.array(all_sites_atom_info)
    all_sites_atom_info = np.expand_dims(all_sites_atom_info, axis=1)
    all_sites_atom_info = np.tile(all_sites_atom_info, [1, max_neighbor_num, 1])

    neighbor_atom_info = []
    for each_neighbor in neighbor_atom_num:
        neighbor_atom_info.append([atom_info[num - 1] for num in each_neighbor])
    neighbor_atom_info = np.array(neighbor_atom_info)

    features =(all_sites_atom_info, neighbor_atom_info, onehot_neighbor_distance)
    encoded_structure = np.concatenate(features, axis=2)

    # shape=   (num_sites)      (num_sites, max_neighbor_num)    (num_sites, max_neighbor_num, onehot_length)
    #   ex.    (8, 10, 47)      (8, 10, 47)                      (8, 10, 10)
    #return (all_sites_atom_info, neighbor_index, onehot_neighbor_distance)
    return encoded_structure





if __name__ == "__main__":
    radius = 3
    max_neighbor_num = 10
    cifs = ["./Data/" + f for f in os.listdir("./Data") if ".cif" in f]
    structures = [IStructure.from_file(cif) for cif in cifs]
    structure = IStructure.from_file("./Data/from_cgcnn/9009743.cif")
    encoded_structure = structure_encoder(structure, radius, max_neighbor_num)
    print(encoded_structure.shape)
