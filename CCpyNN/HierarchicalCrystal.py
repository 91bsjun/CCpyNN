import os
import json
import numpy as np

from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import IStructure

structure = IStructure.from_file("./Data/from_cgcnn/1000041.cif")
grid_size = 1.0

sites = structure.sites
