import re
import h5py
from tqdm import tqdm
import torch as pt
import numpy as np
from glob import glob
from src.dataset import StructuresDataset, save
from src.structure import cle_structure, split_by_chain, delete_hetatm, fil_non_atomic_sub, rm_duplicate_tagged_sub, integrate_protein
from src.data_encoding import en_structure, ext_topology, ext_all_contacts, config_encoding, en_features

config = {
    "r_thr": 5.0,
    "max_num_nn": 64,
    "max_num_atoms": 1024*32,
    "molecule_ids": np.array([
        'GLU', 'LEU', 'ALA', 'ASP', 'SER', 'VAL', 'GLY', 'THR', 'ARG', 'PHE', 'TYR', 'ILE',
        'PRO', 'ASN', 'LYS', 'GLN', 'HIS', 'TRP', 'MET', 'CYS', 'A', 'U', 'G', 'C', 'DA',
        'DT', 'DG', 'DC', 'MG', 'ZN', 'CL', 'CA', 'NA', 'MN', 'K', 'IOD', 'CD', 'CU', 'FE',
        'NI', 'SR', 'BR', 'CO', 'HG', 'SO4', 'NAG', 'PO4', 'EDO', 'ACT', 'MAN', 'HEM', 'FMT',
        'BMA', 'ADP', 'FAD', 'NAD', 'NO3', 'GLC', 'ATP', 'NAP', 'BGC', 'GDP', 'FUC', 'FES',
        'FMN', 'GAL', 'GTP', 'PLP', 'MLI', 'ANP', 'H4B', 'AMP', 'NDP', 'SAH', 'OXY', 'PLM',
        'CLR', 'CDL', 'RET'
    ])
}
pt.multiprocessing.set_sharing_strategy('file_system')

def con_types(sf, Mf, se, Me, ids, molecule_ids, device=pt.device("cpu")):
    c0 = pt.from_numpy(sf['resname'].reshape(-1,1) == molecule_ids.reshape(1,-1)).to(device)
    c1 = pt.from_numpy(se['resname'].reshape(-1,1) == molecule_ids.reshape(1,-1)).to(device)
    H = (c1[ids[:,1]].unsqueeze(1) & c0[ids[:,0]].unsqueeze(2))
    rids0 = pt.where(Mf[ids[:,0]])[1]
    rids1 = pt.where(Me[ids[:,1]])[1]
    Y = pt.zeros((Mf.shape[1], Me.shape[1], H.shape[1], H.shape[2]), device=device, dtype=pt.bool)
    Y[rids0, rids1] = H
    T = pt.any(pt.any(Y, dim=1), dim=0)
    return Y, T

def pack_stru_data(X, qe, qr, qn, M, ids_topk):
    return {
        'X': X.cpu().numpy().astype(np.float32),
        'ids_topk': ids_topk.cpu().numpy().astype(np.uint16),
        'qe':pt.stack(pt.where(qe > 0.5), dim=1).cpu().numpy().astype(np.uint16),
        'qr':pt.stack(pt.where(qr > 0.5), dim=1).cpu().numpy().astype(np.uint16),
        'qn':pt.stack(pt.where(qn > 0.5), dim=1).cpu().numpy().astype(np.uint16),
        'M':pt.stack(pt.where(M), dim=1).cpu().numpy().astype(np.uint16),
    }, {
        'qe_shape': qe.shape, 'qr_shape': qr.shape, 'qn_shape': qn.shape,
        'M_shape': M.shape,
    }

def pack_con_data(Y, T):
    return {
        'Y':pt.stack(pt.where(Y), dim=1).cpu().numpy().astype(np.uint16),
    }, {
        'Y_shape': Y.shape, 'ctype': T.cpu().numpy(),
    }

def pack_dataset(subunits, contacts, molecule_ids, max_num_nn, device=pt.device("cpu")):
    structures_data = {}
    contacts_data = {}
    for cid0 in contacts:
        s0 = subunits[cid0]
        qe0, qr0, qn0 = en_features(s0)
        X0, M0 = en_structure(s0, device=device)
        ids0_topk = ext_topology(X0, max_num_nn)[0]
        structures_data[cid0] = pack_stru_data(X0, qe0, qr0, qn0, M0, ids0_topk)
        if cid0 not in contacts_data:
            contacts_data[cid0] = {}
        for cid1 in contacts[cid0]:
            if cid1 not in contacts_data:
                contacts_data[cid1] = {}
            if cid1 not in contacts_data[cid0]:
                s1 = subunits[cid1]
                X1, M1 = en_structure(s1, device=device)
                if (M0.shape[1] * M1.shape[1] * (molecule_ids.shape[0]**2)) > 2e9:
                    ctc_ids = contacts[cid0][cid1]['ids'].cpu()
                    Y, T = con_types(s0, M0.cpu(), s1, M1.cpu(), ctc_ids, molecule_ids, device=pt.device("cpu"))
                else:
                    ctc_ids = contacts[cid0][cid1]['ids'].to(device)
                    Y, T = con_types(s0, M0.to(device), s1, M1.to(device), ctc_ids, molecule_ids, device=device)
                if pt.any(Y):
                    contacts_data[cid0][cid1] = pack_con_data(Y, T)
                    contacts_data[cid1][cid0] = pack_con_data(Y.permute(1,0,3,2), T.transpose(0,1))
                pt.cuda.empty_cache()
    return structures_data, contacts_data

def store_dataset(hf, pdbid, bid, structures_data, contacts_data):
    metadata_l = []
    for cid0 in contacts_data:
        key = f"{pdbid.upper()[1:3]}/{pdbid.upper()}/{bid}/{cid0}"
        hgrp = hf.create_group(f"data/structures/{key}")
        save(hgrp, attrs=structures_data[cid0][1], **structures_data[cid0][0])
        for cid1 in contacts_data[cid0]:
            ckey = f"{key}/{cid1}"
            hgrp = hf.create_group(f"data/contacts/{ckey}")
            save(hgrp, attrs=contacts_data[cid0][cid1][1], **contacts_data[cid0][cid1][0])
            metadata_l.append({
                'key': key,
                'size': (np.max(structures_data[cid0][0]["M"], axis=0)+1).astype(int),
                'ckey': ckey,
                'ctype': contacts_data[cid0][cid1][1]["ctype"],
            })
    return metadata_l

def main(complex_type):
    if complex_type == 'protein':
        pdb_filepaths = glob("data/scannet/*.pdb[0-9].gz")
        dataset_filepath = "data/scannet.h5"
    elif complex_type == 'peptide':
        pdb_filepaths = glob("data/pepnn/*.pdb[0-9].gz")
        dataset_filepath = "data/pepnn.h5"
    dataset = StructuresDataset(pdb_filepaths, with_preprocessing=False)
    dataloader = pt.utils.data.DataLoader(dataset, batch_size=None, shuffle=True, num_workers=16, pin_memory=False, prefetch_factor=4)
    device = pt.device("cuda:1")
    with h5py.File(dataset_filepath, 'w', libver='latest') as hf:
        for key in config_encoding:
            hf[f"metadata/{key}"] = config_encoding[key].astype(np.string_)
        hf["metadata/mids"] = config['molecule_ids'].astype(np.string_)
        metadata_l = []
        pbar = tqdm(dataloader)
        for structure, pdb_filepath in pbar:
            if structure is None:
                continue
            m = re.match(r'.*/([a-z0-9]*)\.pdb([0-9]*)\.gz', pdb_filepath)
            pdbid = m[1]
            bid = m[2]
            if structure['xyz'].shape[0] >= config['max_num_atoms']:
                continue
            structure = cle_structure(structure)
            structure = delete_hetatm(structure)
            subunits = split_by_chain(structure)
            if type == 'peptide':
                subunits = integrate_protein(subunits)
            subunits = fil_non_atomic_sub(subunits)
            if len(subunits) < 2:
                continue
            subunits = rm_duplicate_tagged_sub(subunits)
            contacts = ext_all_contacts(subunits, config['r_thr'], device=device)
            if len(contacts) == 0:
                continue
            structures_data, contacts_data = pack_dataset(
                subunits, contacts,
                config['molecule_ids'],
                config['max_num_nn'], device=device
            )
            metadata = store_dataset(hf, pdbid, bid, structures_data, contacts_data)
            metadata_l.extend(metadata)
            pbar.set_description(f"{metadata_l[-1]['key']}: {metadata_l[-1]['size']}")
        hf['metadata/keys'] = np.array([m['key'] for m in metadata_l]).astype(np.string_)
        hf['metadata/sizes'] = np.array([m['size'] for m in metadata_l])
        hf['metadata/ckeys'] = np.array([m['ckey'] for m in metadata_l]).astype(np.string_)
        hf['metadata/ctypes'] = np.stack(np.where(np.array([m['ctype'] for m in metadata_l])), axis=1).astype(np.uint32)

if __name__ == "__main__":
    main(complex_type='peptide')