import itertools
import numpy as np
import torch
import hydra

from scipy.spatial.distance import pdist
from scipy.spatial.distance import cdist
from hydra.experimental import compose
from hydra import initialize_config_dir
from pathlib import Path

import smact
from smact.screening import pauling_test

from torch_geometric.data import DataLoader

# CompScaler = StandardScaler(
#     means=np.array(CompScalerMeans),
#     stds=np.array(CompScalerStds),
#     replace_nan_token=0.)


# def smact_validity(comp, count,
#                    use_pauling_test=True,
#                    include_alloys=True):
#     elem_symbols = tuple([chemical_symbols[elem] for elem in comp])
#     space = smact.element_dictionary(elem_symbols)
#     smact_elems = [e[1] for e in space.items()]
#     electronegs = [e.pauling_eneg for e in smact_elems]
#     ox_combos = [e.oxidation_states for e in smact_elems]
#     if len(set(elem_symbols)) == 1:
#         return True
#     if include_alloys:
#         is_metal_list = [elem_s in smact.metals for elem_s in elem_symbols]
#         if all(is_metal_list):
#             return True

#     threshold = np.max(count)
#     compositions = []
#     for ox_states in itertools.product(*ox_combos):
#         stoichs = [(c,) for c in count]
#         # Test for charge balance
#         cn_e, cn_r = smact.neutral_ratios(
#             ox_states, stoichs=stoichs, threshold=threshold)
#         # Electronegativity test
#         if cn_e:
#             if use_pauling_test:
#                 try:
#                     electroneg_OK = pauling_test(ox_states, electronegs)
#                 except TypeError:
#                     # if no electronegativity data, assume it is okay
#                     electroneg_OK = True
#             else:
#                 electroneg_OK = True
#             if electroneg_OK:
#                 for ratio in cn_r:
#                     compositions.append(
#                         tuple([elem_symbols, ox_states, ratio]))
#     compositions = [(i[0], i[2]) for i in compositions]
#     compositions = list(set(compositions))
#     if len(compositions) > 0:
#         return True
#     else:
#         return False


def structure_validity(crystal, cutoff=0.5):
    dist_mat = crystal.distance_matrix
    # Pad diagonal with a large number
    dist_mat = dist_mat + np.diag(
        np.ones(dist_mat.shape[0]) * (cutoff + 10.))
    if dist_mat.min() < cutoff or crystal.volume < 0.1:
        return False
    else:
        return True


def get_fp_pdist(fp_array):
    if isinstance(fp_array, list):
        fp_array = np.array(fp_array)
    fp_pdists = pdist(fp_array)
    return fp_pdists.mean()


def filter_fps(struc_fps, comp_fps):
    assert len(struc_fps) == len(comp_fps)

    filtered_struc_fps, filtered_comp_fps = [], []

    for struc_fp, comp_fp in zip(struc_fps, comp_fps):
        if struc_fp is not None and comp_fp is not None:
            filtered_struc_fps.append(struc_fp)
            filtered_comp_fps.append(comp_fp)
    return filtered_struc_fps, filtered_comp_fps


# struct_fp, comp_fp -> structural fingerprint, composition fingerprint
# def compute_cov(crys, gt_crys,
#                 struc_cutoff, comp_cutoff, num_gen_crystals=None):
#     struc_fps = [c.struct_fp for c in crys]
#     comp_fps = [c.comp_fp for c in crys]
#     gt_struc_fps = [c.struct_fp for c in gt_crys]
#     gt_comp_fps = [c.comp_fp for c in gt_crys]

#     assert len(struc_fps) == len(comp_fps)
#     assert len(gt_struc_fps) == len(gt_comp_fps)

#     # Use number of crystal before filtering to compute COV
#     if num_gen_crystals is None:
#         num_gen_crystals = len(struc_fps)

#     struc_fps, comp_fps = filter_fps(struc_fps, comp_fps)

#     comp_fps = CompScaler.transform(comp_fps)
#     gt_comp_fps = CompScaler.transform(gt_comp_fps)

#     struc_fps = np.array(struc_fps)
#     gt_struc_fps = np.array(gt_struc_fps)
#     comp_fps = np.array(comp_fps)
#     gt_comp_fps = np.array(gt_comp_fps)

#     struc_pdist = cdist(struc_fps, gt_struc_fps)
#     comp_pdist = cdist(comp_fps, gt_comp_fps)

#     struc_recall_dist = struc_pdist.min(axis=0)
#     struc_precision_dist = struc_pdist.min(axis=1)
#     comp_recall_dist = comp_pdist.min(axis=0)
#     comp_precision_dist = comp_pdist.min(axis=1)

#     cov_recall = np.mean(np.logical_and(
#         struc_recall_dist <= struc_cutoff,
#         comp_recall_dist <= comp_cutoff))
#     cov_precision = np.sum(np.logical_and(
#         struc_precision_dist <= struc_cutoff,
#         comp_precision_dist <= comp_cutoff)) / num_gen_crystals

#     metrics_dict = {
#         'cov_recall': cov_recall,
#         'cov_precision': cov_precision,
#         'amsd_recall': np.mean(struc_recall_dist),
#         'amsd_precision': np.mean(struc_precision_dist),
#         'amcd_recall': np.mean(comp_recall_dist),
#         'amcd_precision': np.mean(comp_precision_dist),
#     }

#     combined_dist_dict = {
#         'struc_recall_dist': struc_recall_dist.tolist(),
#         'struc_precision_dist': struc_precision_dist.tolist(),
#         'comp_recall_dist': comp_recall_dist.tolist(),
#         'comp_precision_dist': comp_precision_dist.tolist(),
#     }

#     return metrics_dict, combined_dist_dict