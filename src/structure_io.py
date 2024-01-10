import gemmi
import numpy as np

def save_pdb(subunits, filepath):
    with open(filepath, 'w') as fs:
        for cn in subunits:
            N = subunits[cn]['xyz'].shape[0]
            for i in range(N):
                h = "ATOM" if subunits[cn]['het_flag'][i] == 'A' else "HETATM"
                n = subunits[cn]['name'][i]
                rn = subunits[cn]['resname'][i]
                e = subunits[cn]['element'][i]
                ri = subunits[cn]['resid'][i]
                xyz = subunits[cn]['xyz'][i]
                if "bfactor" in subunits[cn]:
                    bf = subunits[cn]['bfactor'][i]
                else:
                    bf = 0.0
                c = cn.split(':')[0][0]
                pdb_line = "{:<6s}{:>5d} {:<4s} {:>3s} {:1s}{:>4d}    {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:<2s}  ".format(h, i+1, n, rn, c, ri, xyz[0], xyz[1], xyz[2], bf, bf, e)
                fs.write(pdb_line+'\n')
            fs.write("TER\n")
        fs.write("END")

def read_pdb(pdb_filepath):
    doc = gemmi.read_pdb(pdb_filepath, max_line_length=80)
    altloc_l = []
    icodes = []
    atom_element = []
    atom_name = []
    atom_xyz = []
    residue_name = []
    seq_id = []
    het_flag = []
    chain_name = []
    model = doc[0]
    mid = 0
    for a in model.all():
        if a.atom.has_altloc():
            key = f"{a.chain.name}_{a.residue.seqid.num}_{a.atom.name}"
            if key in altloc_l:
                continue
            else:
                altloc_l.append(key)
        icodes.append(a.residue.seqid.icode.strip())
        atom_element.append(a.atom.element.name)
        atom_name.append(a.atom.name)
        atom_xyz.append([a.atom.pos.x, a.atom.pos.y, a.atom.pos.z])
        residue_name.append(a.residue.name)
        seq_id.append(a.residue.seqid.num)
        het_flag.append(a.residue.het_flag)
        chain_name.append(f"{a.chain.name}:{mid}")
    return {
        'xyz': np.array(atom_xyz, dtype=np.float32),
        'name': np.array(atom_name),
        'element': np.array(atom_element),
        'resname': np.array(residue_name),
        'resid': np.array(seq_id, dtype=np.int32),
        'het_flag': np.array(het_flag),
        'chain_name': np.array(chain_name),
        'icode': np.array(icodes),
    }