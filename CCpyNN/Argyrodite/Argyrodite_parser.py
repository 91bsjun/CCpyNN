#!/bin/env python
import os, sys
import re
import numpy as np
import pandas as pd

from pymatgen.core.periodic_table import Specie
from pymatgen import Structure



class analyze_4a4c:
    def __init__(self, structure):        
        st = structure
        num_sites = len(st.sites) 
        # -- create dummy sites
        for c_site in c_sites:
            st.append(Specie("Kr"), c_site)
        for a_site in a_sites:
            st.append(Specie("Xe"), a_site)

        c_index = list(range(num_sites, num_sites + len(c_sites)))      # Get dummy sites indice
        a_index = list(range(num_sites + len(c_sites), num_sites + len(c_sites) + len(a_sites)))

        self.st = st
        self.c_index = c_index
        self.a_index = a_index
        self.done_ref_sites = {}
   
    def pre_parsing(self):
        st = self.st
        sites = st.sites
        c_index = self.c_index
        a_index = self.a_index
        
        X_index = [i for i in range(len(sites)) if str(sites[i].specie) in Xs]      # Get X, B sites indice
        O_index = [i for i in range(len(sites)) if str(sites[i].specie) in Os]
        
        len_Xs = len(X_index)
        len_Os = 8 - len(X_index)

        # -- find single O sites
        single_O_index = []
        dist_tol = 0
        tol_val = 0.01
        tol_cnt = 0
        while len(single_O_index) != len_Os:
            #print(tol_cnt, tol_val, dist_tol, len(single_O_index))
            tol_cnt += 1
            if tol_cnt % 1000 == 0:           # decrease tolerance when not converge
                tol_val = tol_val * 0.1
            single_O_index = []
            for i in O_index:
                for j in c_index + a_index:
                    if st.get_distance(i, j) < dist_tol:
                        if i not in single_O_index:
                            single_O_index.append(i)
            if len(single_O_index) > len_Os:
                dist_tol -= tol_val
            else:
                dist_tol += tol_val
#            if tol_cnt == 10000:
#                self.info = {"X in a": 'NaN', "chalcogen in a": 'NaN', "X in c": 'NaN', "chalcogen in c": 'NaN'}
#                return 
        O_index = single_O_index


        info = {"X in a": 0, "chalcogen in a": 0, "X in c": 0, "chalcogen in c": 0}
        anion_site_info = {}            # {site_i: ['c', 'Cl', array([0.257687, 0.285818, 0.797124])]}

        # materials for finding multiple used most close free anion sites
        parsed_c_index = {}
        for ci in c_index:
            parsed_c_index[ci] = []
        parsed_a_index = {}
        for ai in a_index:
            parsed_a_index[ai] = []
            
        # find where X included
        for xi in X_index:
            c_dists = []
            a_dists = []
            for ci in c_index:
                c_dists.append(st.get_distance(xi, ci))
            for ai in a_index:
                a_dists.append(st.get_distance(xi, ai))

            min_dist = min(c_dists + a_dists)
            for ci in c_index:
                if min_dist == st.get_distance(xi, ci):
                    parsed_c_index[ci].append(xi)

            for ai in a_index:
                if min_dist == st.get_distance(xi, ai):
                    parsed_a_index[ai].append(xi)

        # find where chalogen included
        for oi in O_index:
            c_dists = []
            a_dists = []
            for ci in c_index:
                c_dists.append(st.get_distance(oi, ci))
            for ai in a_index:
                a_dists.append(st.get_distance(oi, ai))

            min_dist = min(c_dists + a_dists)
            for ci in c_index:
                if min_dist == st.get_distance(oi, ci):
                    parsed_c_index[ci].append(oi)

            for ai in a_index:
                if min_dist == st.get_distance(oi, ai):
                    parsed_a_index[ai].append(oi)
                
            if min(c_dists) < min(a_dists):
                anion_site_info[oi] = ['c', str(sites[oi].specie), sites[oi].frac_coords]
            else:
                anion_site_info[oi] = ['a', str(sites[oi].specie), sites[oi].frac_coords]

        # c 사이트에 2개 이상 음이온이 할당된 경우 가장 적합한것 1개 제외하고 losts에 추가
        losts = []
        for ci in c_index:
            if len(parsed_c_index[ci]) >= 2:
                dists = [st.get_distance(ci, i) for i in parsed_c_index[ci]]
                min_dist = min(dists)
                for i in parsed_c_index[ci]:
                    if min_dist != st.get_distance(ci, i):
                        losts.append(i)
                    else:
                        parsed_c_index[ci] = [i]
        # 할당이 안된 a site 에 losts 추가
        for ai in a_index:
            if len(parsed_a_index[ai]) == 0:
                dists = [st.get_distance(ai, i) for i in losts]
                for i in losts:
                    if min(dists) == st.get_distance(ai, i):
                        parsed_a_index[ai] = [i]
        # 할당이 안된 c site 에 losts 추가
        for ci in c_index:
            if len(parsed_c_index[ci]) == 0:
                dists = [st.get_distance(ci, i) for i in losts]
                for i in losts:
                    if min(dists) == st.get_distance(ci, i):
                        parsed_c_index[ci] = [i]

        # 반대경우 반복
        losts = []
        for ai in a_index:
            if len(parsed_a_index[ai]) >= 2:
                dists = [st.get_distance(ai, i) for i in parsed_a_index[ai]]
                min_dist = min(dists)
                for i in parsed_a_index[ai]:
                    if min_dist != st.get_distance(ai, i):
                        losts.append(i)
                    else:
                        parsed_a_index[ai] = [i]
        for ci in c_index:
            if len(parsed_c_index[ci]) == 0:
                dists = [st.get_distance(ci, i) for i in losts]
                for i in losts:
                    if min(dists) == st.get_distance(ci, i):
                        parsed_c_index[ci] = [i]
        for ai in a_index:
            if len(parsed_a_index[ai]) == 0:
                dists = [st.get_distance(ai, i) for i in losts]
                for i in losts:
                    if min(dists) == st.get_distance(ai, i):
                        parsed_a_index[ai] = [i]

        empty_keys = []
        for key in parsed_a_index.keys():
            if len(parsed_a_index[key]) == 0:
                empty_keys.append(key)
        for key in empty_keys:
            del parsed_a_index[key]

                                        # parsed_a_index = {56: [48], 57: [49], 58: [50], 59: [28]}
        for ai in parsed_a_index:       # parsed_a_index 의 key는 기본 a 사이트이고, val 은 해당 사이트에 매칭시킨 실제 구조의 free anion 임
            val = parsed_a_index[ai][0]
            if str(sites[val].specie) in Xs:
                info["X in a"] += 1
                anion_site_info[val] = ['a', str(sites[val].specie), sites[val].frac_coords]
            else:
                info["chalcogen in a"] += 1
                anion_site_info[val] = ['a', str(sites[val].specie), sites[val].frac_coords]
        for ci in parsed_c_index:       # parsed_a_index 의 key는 기본 a 사이트이고, val 은 해당 사이트에 매칭시킨 실제 구조의 free anion 임
            val = parsed_c_index[ci][0]
            if str(sites[val].specie) in Xs:
                info["X in c"] += 1
                anion_site_info[val] = ['c', str(sites[val].specie), sites[val].frac_coords]
            else:
                info["chalcogen in c"] += 1
                anion_site_info[val] = ['c', str(sites[val].specie), sites[val].frac_coords]

        self.anion_site_info = anion_site_info
        self.info = info
        self.X_index = X_index
        self.O_index = O_index

def analyze_4a4c_from_filename(filename):
    if '4a' in filename or '4004' in filename:
        return [4, 0, 0, 4]
    elif '3a1c' in filename or '3113' in filename:
        return [3, 1, 1, 3]
    elif '2a2c' in filename or '2222' in filename:
        return [2, 2, 2, 2]
    elif '1a3c' in filename or '1331' in filename:
        return [1, 3, 3, 1]
    elif '4c' in filename or '0440' in filename:
        return [0, 4, 4, 0]
    else:
        return False

def info_arrhenius(filename):
    """
    data:
        Name,
        Ea (eV),
        Ea_err (+/-),
        ext_D (cm^2/s),
        D_err_from,
        D_err_to,
        ext_c (mS/cm),
        c_err_from,
        c_err_to
    """
    df = pd.read_csv(filename)
    data = df.to_dict(orient='index')

    return data[0]


c_sites = [[0.25, 0.25, 0.75], [0.75, 0.75, 0.75], [0.75, 0.25, 0.25], [0.25, 0.75, 0.25]]
a_sites = [[0., 0., 0.5], [0.5, 0.5, 0.5], [0., 0.5, 0.], [0.5, 0., 0.]]
a_sites_extra = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.0], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0], [0.5, 0.5, 0.5]]
a_sites = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.0], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0], [0.5, 0.5, 0.5]]
Xs = ['F', 'Cl', 'Br', 'I']
Os = ['O', 'S', 'Se', 'Te']
polys = ['P', 'Sb', 'Si', 'Ge', 'Sn']

if __name__ == "__main__":
    data = {"composition": [],
            "num_of_li": [],
            "4b_1":[], "4b_2":[], "4b_3": [], "4b_4": [],
            "16e": [],
            "4a_1": [], "4a_2": [], "4a_3": [], "4a_4": [],
            "4c_1": [], "4c_2": [], "4c_3": [], "4c_4": [],
            'volume': [],
            'conductivity': []}
    poly_anion_keys = ["4b_1", "4b_2", "4b_3", "4b_4"]
    free_anion_4a_keys = ["4a_1", "4a_2", "4a_3", "4a_4"]
    free_anion_4c_keys = ["4c_1", "4c_2", "4c_3", "4c_4"]

    dirs = [d for d in os.listdir()]
    dirs.sort()

    from CCpy.Tools.CCpyTools import progress_bar
    crt = 0
    for d in dirs:
        crt += 1
        progress_bar(len(dirs), crt, len_bar=50, cmt=d)
        os.chdir(d)
        cif_files = [f for f in os.listdir() if '.cif' in f]
        structure = Structure.from_file(cif_files[0])

        composition = structure.composition
        data['composition'].append(str(composition.formula).replace(" ",""))
        
        num_of_li = len([s for s in structure.sites if str(s.specie) == "Li"])
        data["num_of_li"].append(num_of_li)

        poly_anions = [s for s in structure.sites if str(s.specie) in polys]        
        for i, pa in enumerate(poly_anions):
            data[poly_anion_keys[i]].append(str(pa.specie))

        for elt in structure.symbol_set:
            if elt in Os:
                data["16e"].append(elt)
                break

        analyzer = analyze_4a4c(structure)
        analyzer.pre_parsing()
        anion_site_info = analyzer.anion_site_info

        site_4a = []
        site_4c = []
        for key in anion_site_info.keys():
            if anion_site_info[key][0] == 'a':
                site_4a.append(anion_site_info[key][1])
            elif anion_site_info[key][0] == 'c':
                site_4c.append(anion_site_info[key][1])

        for i in range(4):
            data[free_anion_4a_keys[i]].append(site_4a[i])
            data[free_anion_4c_keys[i]].append(site_4c[i])

        data['volume'].append(structure.volume)

        arrhenius_data = info_arrhenius('arrhenius_fit.csv')
        data['conductivity'].append(arrhenius_data['ext_c (mS/cm)'])              # 7
        os.chdir('../')

    df = pd.DataFrame(data)
    print(df)
    df.to_csv("sites_info_data.csv")
