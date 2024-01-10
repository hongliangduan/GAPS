import torch as pt
from torch.utils.checkpoint import checkpoint

def unpack_state_features(X, ids_topk, q):
    R_nn = X[ids_topk-1] - X.unsqueeze(1)
    D_nn = pt.norm(R_nn, dim=2)
    D_nn = D_nn + pt.max(D_nn)*(D_nn < 1e-2).float()
    R_nn = R_nn / D_nn.unsqueeze(2)
    q = pt.cat([pt.zeros((1, q.shape[1]), device=q.device), q], dim=0)
    ids_topk = pt.cat([pt.zeros((1, ids_topk.shape[1]), dtype=pt.long, device=ids_topk.device), ids_topk], dim=0)
    D_nn = pt.cat([pt.zeros((1, D_nn.shape[1]), device=D_nn.device), D_nn], dim=0)
    R_nn = pt.cat([pt.zeros((1, R_nn.shape[1], R_nn.shape[2]), device=R_nn.device), R_nn], dim=0)
    return q, ids_topk, D_nn, R_nn

class StateUpdate(pt.nn.Module):
    def __init__(self, Ns, Nh, Nk):
        super(StateUpdate, self).__init__()
        self.Ns = Ns
        self.Nh = Nh
        self.Nk = Nk
        self.nqm = pt.nn.Sequential(
            pt.nn.Linear(2*Ns, Ns),
            pt.nn.ELU(),
            pt.nn.Linear(Ns, Ns),
            pt.nn.ELU(),
            pt.nn.Linear(Ns, 2*Nk*Nh),
        )
        self.epkm = pt.nn.Sequential(
            pt.nn.Linear(6*Ns+1, Ns),
            pt.nn.ELU(),
            pt.nn.Linear(Ns, Ns),
            pt.nn.ELU(),
            pt.nn.Linear(Ns, 3*Nk),
        )
        self.evm = pt.nn.Sequential(
            pt.nn.Linear(6*Ns+1, 2*Ns),
            pt.nn.ELU(),
            pt.nn.Linear(2*Ns, 2*Ns),
            pt.nn.ELU(),
            pt.nn.Linear(2*Ns, Ns),
        )
        self.qpm = pt.nn.Sequential(
            pt.nn.Linear(Nh*Ns, Ns),
            pt.nn.ELU(),
            pt.nn.Linear(Ns, Ns),
            pt.nn.ELU(),
            pt.nn.Linear(Ns, Ns),
        )
        self.ppm = pt.nn.Sequential(
            pt.nn.Linear(Nh*Ns, Ns, bias=False),
        )
        self.sdk = pt.nn.Parameter(pt.sqrt(pt.tensor(Nk).float()), requires_grad=False)
        self.nkm = pt.nn.Sequential(
            pt.nn.Linear(3*Ns, Ns),
            pt.nn.ELU(),
            pt.nn.Linear(Ns, Ns),
            pt.nn.ELU(),
            pt.nn.Linear(Ns, Nk),
        )
        self.nvm = pt.nn.Sequential(
            pt.nn.Linear(3*Ns, Ns),
            pt.nn.ELU(),
            pt.nn.Linear(Ns, Ns),
            pt.nn.ELU(),
            pt.nn.Linear(Ns, Ns),
        )

    def forward(self, q, p, q_nn, p_nn, d_nn, r_nn):
        N, n, S = q_nn.shape
        X_n = pt.cat([
            q,
            pt.norm(p, dim=1),
        ], dim=1) 
        X_c = pt.cat([q_nn, pt.norm(p_nn, dim=2), pt.sum(p_nn * r_nn.unsqueeze(3), dim=2)], dim=2) 
        X_e = pt.cat([
            d_nn.unsqueeze(2),                                 
            X_n.unsqueeze(1).repeat(1,n,1),                    
            q_nn,                                             
            pt.norm(p_nn, dim=2),                              
            pt.sum(p.unsqueeze(1) * r_nn.unsqueeze(3), dim=2),  
            pt.sum(p_nn * r_nn.unsqueeze(3), dim=2),           
        ], dim=2)  
        Q = self.nqm.forward(X_n).view(N, 2, self.Nh, self.Nk)  
        Ke = pt.cat(pt.split(self.epkm.forward(X_e), self.Nk, dim=2), dim=1).transpose(1,2) 
        V = self.evm.forward(X_e)  
        Ve = pt.cat([V.unsqueeze(2) * r_nn.unsqueeze(3),p.unsqueeze(1).repeat(1,n,1,1),p_nn], dim=1).transpose(1,2) 
        Kn = self.nkm.forward(X_c).transpose(1,2) 
        Vn = self.nvm.forward(X_c) 
        Mq = pt.nn.functional.softmax(pt.matmul(Q[:,0], Kn) / self.sdk, dim=2)  
        Mp = pt.nn.functional.softmax(pt.matmul(Q[:,1], Ke) / self.sdk, dim=2)  
        Zq = pt.matmul(Mq, Vn).view(N, self.Nh*self.Ns) 
        Zp = pt.matmul(Mp.unsqueeze(1), Ve).view(N, 3, self.Nh*self.Ns) 
        qh = self.qpm.forward(Zq)
        ph = self.ppm.forward(Zp)
        qz = q + qh
        pz = p + ph
        return qz, pz

class StatePoolLayer(pt.nn.Module):
    def __init__(self, N0, N1, Nh):
        super(StatePoolLayer, self).__init__()
        self.sam = pt.nn.Sequential(
            pt.nn.Linear(2*N0, N0),
            pt.nn.ELU(),
            pt.nn.Linear(N0, N0),
            pt.nn.ELU(),
            pt.nn.Linear(N0, 2*Nh),
        )
        self.zdm = pt.nn.Sequential(
            pt.nn.Linear(Nh * N0, N0),
            pt.nn.ELU(),
            pt.nn.Linear(N0, N0),
            pt.nn.ELU(),
            pt.nn.Linear(N0, N1),
        )
        self.zdm_vec = pt.nn.Sequential(
            pt.nn.Linear(Nh * N0, N1, bias=False)
        )

    def forward(self, q, p, M):
        F = (1.0 - M + 1e-6) / (M - 1e-6)
        z = pt.cat([q, pt.norm(p, dim=1)], dim=1)
        Ms = pt.nn.functional.softmax(self.sam.forward(z).unsqueeze(1) + F.unsqueeze(2), dim=0).view(M.shape[0], M.shape[1], -1, 2)
        qh = pt.matmul(pt.transpose(q,0,1), pt.transpose(Ms[:,:,:,0],0,1))
        ph = pt.matmul(pt.transpose(pt.transpose(p,0,2),0,1), pt.transpose(Ms[:,:,:,1],0,1).unsqueeze(1))
        qr = self.zdm.forward(qh.view(Ms.shape[1], -1))
        pr = self.zdm_vec.forward(ph.view(Ms.shape[1], p.shape[1], -1))
        return qr, pr


class StateUpdateLayer(pt.nn.Module):
    def __init__(self, layer_params):
        super(StateUpdateLayer, self).__init__()
        self.su = StateUpdate(*[layer_params[k] for k in ['Ns', 'Nh', 'Nk']])
        self.m_nn = pt.nn.Parameter(pt.arange(layer_params['nn'], dtype=pt.int64), requires_grad=False)


    def forward(self, Z):
        q, p, ids_topk, D_topk, R_topk = Z
        ids_nn = ids_topk[:,self.m_nn]
        q = q.requires_grad_()
        p = p.requires_grad_()
        q, p = checkpoint(self.su.forward, q, p, q[ids_nn], p[ids_nn], D_topk[:,self.m_nn], R_topk[:,self.m_nn])
        q[0] = q[0] * 0.0
        p[0] = p[0] * 0.0
        return q, p, ids_topk, D_topk, R_topk