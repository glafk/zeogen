from collections import Counter
import numpy as np

from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition
from pymatgen.core.lattice import Lattice

from evaluation.eval_utils import structure_validity
"""
This code snippet imports featurizers for site fingerprint and composition, then
initializes specific fingerprint presets for crystal structures and elemental
properties.

1. `CrystalNNFingerprint` from `matminer.featurizers.site.fingerprint` is a
   featurizer that generates a fingerprint representation of a crystal
   structure. It uses a neural network to learn a mapping from crystal
   structures to a compact vector representation, which can be used for tasks
   like classification or clustering. In this code, it is initialized with the
   "ops" preset, which stands for "only periodic properties".

2. `ElementProperty` from `matminer.featurizers.composition.composite` is a
   featurizer that extracts elemental properties from an elemental composition.
   It uses a set of predefined properties such as atomic mass,
   electronegativity, ionization energy, and electron affinity. In this code, it
   is initialized with the "magpie" preset, which stands for "Magpie's
   properties".

These featurizers can be used in various machine learning tasks, such as
predicting physical or chemical properties of materials, identifying patterns in
crystal structures, or clustering similar materials. They provide a way to
represent complex data (crystal structures and elemental compositions) in a more
manageable and comparable form.
"""

from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
from matminer.featurizers.composition.composite import ElementProperty

CrystalNNFP = CrystalNNFingerprint.from_preset("ops")
CompFP = ElementProperty.from_preset('magpie')

class Crystal(object):

    def __init__(self, crys_dict):
        self.frac_coords = crys_dict['frac_coords'].detach().cpu()
        self.atom_types = crys_dict['atom_types'].detach().cpu()
        self.lengths = crys_dict['lengths'].detach().cpu()
        self.angles = crys_dict['angles'].detach().cpu()
        self.dict = crys_dict

        self.get_structure()
        self.get_composition()
        self.get_validity()
        self.get_fingerprints()

    def get_structure(self):
        self.lengths = self.lengths.tolist()[0]
        self.angles = self.angles.tolist()[0]
        if min(self.lengths) < 0:
            self.constructed = False
            self.invalid_reason = 'non_positive_lattice'
        else:
            try:
                self.structure = Structure(
                    lattice=Lattice.from_parameters(
                        *(self.lengths + self.angles)),
                    species=self.atom_types, coords=self.frac_coords, coords_are_cartesian=False)
                self.constructed = True
            except Exception as e:
                print(e)
                self.constructed = False
                self.invalid_reason = 'construction_raises_exception'
            if self.structure.volume < 0.1:
                self.constructed = False
                self.invalid_reason = 'unrealistically_small_lattice'

    def get_composition(self):
        elem_counter = Counter(self.atom_types)
        composition = [(elem, elem_counter[elem])
                       for elem in sorted(elem_counter.keys())]
        elems, counts = list(zip(*composition))
        counts = np.array(counts)
        counts = counts / np.gcd.reduce(counts)
        self.elems = elems
        self.comps = tuple(counts.astype('int').tolist())

    def get_validity(self):
        # Question? Do we need smact validity if we don't care about charge neutrality
        # self.comp_valid = smact_validity(self.elems, self.comps)
        # Getting rid of smact validity for now. Not very applicable in this case
        if self.constructed:
            self.struct_valid = structure_validity(self.structure)
        else:
            self.struct_valid = False
        self.valid = self.struct_valid
        # self.valid = self.comp_valid and self.struct_valid

    def get_fingerprints(self):
        """
        Generates fingerprints for the crystal structure.

        This function calculates the fingerprints for the crystal structure by performing the following steps:
        1. Counts the occurrences of each element in the atom types list using the `Counter` class.
        2. Creates a composition object from the element counts.
        3. Generates the composition fingerprint using the `CompFP.featurize` method.
        4. Tries to generate the site fingerprints for each site in the crystal structure using the `CrystalNNFP.featurize` method.
           If an exception occurs during the fingerprint generation, the crystal is marked as invalid and the fingerprints are set to None.
        5. Calculates the mean of the site fingerprints to obtain the structure fingerprint.

        Returns:
            None
        """
        elem_counter = Counter(self.atom_types)
        comp = Composition(elem_counter)
        self.comp_fp = CompFP.featurize(comp)
        try:
            site_fps = [CrystalNNFP.featurize(
                self.structure, i) for i in range(len(self.structure))]
        except Exception:
            # counts crystal as invalid if fingerprint cannot be constructed.
            self.valid = False
            self.comp_fp = None
            self.struct_fp = None
            return
        self.struct_fp = np.array(site_fps).mean(axis=0)