import numpy as np
import torch as pt
from .structure_io import read_pdb
from .structure import cle_structure, split_by_chain, fil_non_atomic_sub, rm_duplicate_tagged_sub, delete_hetatm

def sele_max_ba(dataset, max_ba):
    aids = np.array([int(key.split('/')[2]) for key in dataset.keys])
    m = (aids <= max_ba)
    return m

def sele_interface_types(dataset, l_types, r_types):
    t0 = np.where(np.isin(dataset.mids, l_types))[0]
    t1 = np.where(np.isin(dataset.mids, r_types))[0]
    cm = (np.isin(dataset.ctypes[:,1], t0) & np.isin(dataset.ctypes[:,2], t1))
    m = np.isin(np.arange(dataset.keys.shape[0]), dataset.ctypes[cm,0])
    return m

def sele_sid(dataset, sids_sel):
    sids = np.array(['_'.join([s.split(':')[0] for s in key.split('/')[1::2]]) for key in dataset.keys])
    m = np.isin(sids, sids_sel)
    return m

def sele_assemblies(dataset, m):
    rmkeys = np.unique(dataset.keys[~m])
    return ~np.isin(dataset.rkeys, rmkeys)

def load(hgrp, keys=None):
    if keys is None:
        keys = hgrp.keys()
    data = {}
    for key in keys:
        data[key] = np.array(hgrp[key])
    attrs = {}
    for key in hgrp.attrs:
        attrs[key] = hgrp.attrs[key]
    return data, attrs

def load_mask(hgrp, k):
    shape = tuple(hgrp.attrs[k+'_shape'])
    M = pt.zeros(shape, dtype=pt.float)
    ids = pt.from_numpy(np.array(hgrp[k]).astype(np.int64))
    M.scatter_(1, ids[:,1:], 1.0)
    return M

def save(hgrp, attrs={}, **data):
    for key in data:
        hgrp.create_dataset(key, data=data[key], compression="lzf")
    for key in attrs:
        hgrp.attrs[key] = attrs[key]


def col_batch(batch_data, max_num_nn=64):
    X = pt.cat([data[0] for data in batch_data], dim=0)
    q = pt.cat([data[2] for data in batch_data], dim=0)
    sizes = pt.tensor([data[3].shape for data in batch_data])
    ids_topk = pt.zeros((X.shape[0], max_num_nn), dtype=pt.long, device=X.device)
    M = pt.zeros(pt.Size(pt.sum(sizes, dim=0)), dtype=pt.float, device=X.device)
    for size, data in zip(pt.cumsum(sizes, dim=0), batch_data):
        ix1 = size[0]
        ix0 = ix1-data[3].shape[0]
        iy1 = size[1]
        iy0 = iy1-data[3].shape[1]
        ids_topk[ix0:ix1, :data[1].shape[1]] = data[1]+ix0+1
        M[ix0:ix1,iy0:iy1] = data[3]
    return X, ids_topk, q, M


class StructuresDataset(pt.utils.data.Dataset):
    def __init__(self, pdb_filepaths, with_preprocessing=True):
        super(StructuresDataset).__init__()
        self.pdb_filepaths = pdb_filepaths
        self.with_preprocessing = with_preprocessing

    def __len__(self):
        return len(self.pdb_filepaths)

    def __getitem__(self, i):
        pdb_filepath = self.pdb_filepaths[i]
        try:
            structure = read_pdb(pdb_filepath)
        except Exception as e:
            print(f"ReadError: {pdb_filepath}: {e}")
            return None, pdb_filepath

        if self.with_preprocessing:
            structure = cle_structure(structure)
            structure = delete_hetatm(structure)
            subunits = split_by_chain(structure)
            subunits = fil_non_atomic_sub(subunits)
            subunits = rm_duplicate_tagged_sub(subunits)
            return subunits, pdb_filepath
        else:
            return structure, pdb_filepath
