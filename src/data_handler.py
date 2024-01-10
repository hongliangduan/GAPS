import h5py
import numpy as np
import torch as pt
from .dataset import load_mask, col_batch

def collate_batch_data(batch_data):
    X, ids_topk, q, M = col_batch(batch_data)
    y = pt.cat([data[4] for data in batch_data])
    return X, ids_topk, q, M, y

def labels(hgrp, t0, t1_l):
    shape = tuple(hgrp.attrs['Y_shape'])
    ids = pt.from_numpy(np.array(hgrp['Y']).astype(np.int64))
    y_ctc_r = pt.any((ids[:,2].view(-1,1) == t0), dim=1).view(-1,1)
    y_ctc_l = pt.stack([pt.any((ids[:,3].view(-1,1) == t1), dim=1) for t1 in t1_l], dim=1)
    y_ctc = (y_ctc_r & y_ctc_l)
    y = pt.zeros((shape[0], len(t1_l)), dtype=pt.bool)
    y[ids[:,0], pt.where(y_ctc)[1]] = True
    return y

class Dataset(pt.utils.data.Dataset):
    def __init__(self, dataset_filepath, features_flags=(True, False, False)):
        super(Dataset, self).__init__()
        self.dataset_filepath = dataset_filepath
        self.ftrs = [fn for fn, ff in zip(['qe','qr','qn'], features_flags) if ff]
        with h5py.File(dataset_filepath, 'r') as hf:
            self.keys = np.array(hf["metadata/keys"]).astype(np.dtype('U'))
            self.sizes = np.array(hf["metadata/sizes"])
            self.ckeys = np.array(hf["metadata/ckeys"]).astype(np.dtype('U'))
            self.ctypes = np.array(hf["metadata/ctypes"])
            self.std_elements = np.array(hf["metadata/std_elements"]).astype(np.dtype('U'))
            self.std_resnames = np.array(hf["metadata/std_resnames"]).astype(np.dtype('U'))
            self.std_names = np.array(hf["metadata/std_names"]).astype(np.dtype('U'))
            self.mids = np.array(hf["metadata/mids"]).astype(np.dtype('U'))
        self.m = np.ones(len(self.keys), dtype=bool)
        self.__update_selection()
        self.t0 = pt.arange(self.mids.shape[0])
        self.t1_l = [pt.arange(self.mids.shape[0])]

    def __update_selection(self):
        self.ckeys_map = {}
        for key, ckey in zip(self.keys[self.m], self.ckeys[self.m]):
            if key in self.ckeys_map:
                self.ckeys_map[key].append(ckey)
            else:
                self.ckeys_map[key] = [ckey]
        self.ukeys = list(self.ckeys_map)

    def update_mask(self, m):
        self.m &= m
        self.__update_selection()

    def set_types(self, l_types, r_types_l):
        self.t0 = pt.from_numpy(np.where(np.isin(self.mids, l_types))[0])
        self.t1_l = [pt.from_numpy(np.where(np.isin(self.mids, r_types))[0]) for r_types in r_types_l]

    def get_largest(self):
        i = np.argmax(self.sizes[:,0] * self.m.astype(int))
        k = np.where(np.isin(self.ukeys, self.keys[i]))[0][0]
        return self[k]

    def __len__(self):
        return len(self.ukeys)

    def __getitem__(self, k):
        key = self.ukeys[k]
        ckeys = self.ckeys_map[key]

        with h5py.File(self.dataset_filepath, 'r') as hf:
            hgrp = hf['data/structures/'+key]
            X = pt.from_numpy(np.array(hgrp['X']).astype(np.float32))
            M = load_mask(hgrp, 'M')
            ids_topk = pt.from_numpy(np.array(hgrp['ids_topk']).astype(np.int64))
            q_l = []
            for fn in self.ftrs:
                q_l.append(load_mask(hgrp, fn))
            q = pt.cat(q_l, dim=1)
            y = pt.zeros((M.shape[1], len(self.t1_l)), dtype=pt.bool)
            for ckey in ckeys:
                y |= labels(hf['data/contacts/'+ckey], self.t0, self.t1_l)

        return X, ids_topk, q, M, y.float()
