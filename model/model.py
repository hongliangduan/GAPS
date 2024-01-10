import torch as pt
from src.model_operations import StateUpdateLayer, StatePoolLayer, unpack_state_features

class Model(pt.nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.em = pt.nn.Sequential(
            pt.nn.Linear(config['em']['N0'], config['em']['N1']),
            pt.nn.ELU(),
            pt.nn.Linear(config['em']['N1'], config['em']['N1']),
            pt.nn.ELU(),
            pt.nn.Linear(config['em']['N1'], config['em']['N1']),
        )
        self.sum = pt.nn.Sequential(*[StateUpdateLayer(layer_params) for layer_params in config['sum']])
        self.spl = StatePoolLayer(config['spl']['N0'], config['spl']['N1'], config['spl']['Nh'])
        self.dm = pt.nn.Sequential(
            pt.nn.Linear(2*config['dm']['N0'], config['dm']['N1']),
            pt.nn.ELU(),
            pt.nn.Linear(config['dm']['N1'], config['dm']['N1']),
            pt.nn.ELU(),
            pt.nn.Linear(config['dm']['N1'], config['dm']['N2']),
        )
        self.transformer_layer = pt.nn.TransformerEncoderLayer(2*config['dm']['N0'], config['transformer_encoder']['head'])
        self.transformer_encoder = pt.nn.TransformerEncoder(self.transformer_layer, config['transformer_encoder']['layer'])
        Nd = config['em']['N1']
        self.qrk = pt.nn.Sequential(
            pt.nn.Linear(Nd, Nd),
            pt.nn.ELU(),
            pt.nn.Linear(Nd, Nd),
            pt.nn.ELU(),
            pt.nn.Linear(Nd, Nd),
        )
        self.qrv = pt.nn.Sequential(
            pt.nn.Linear(Nd, Nd),
            pt.nn.ELU(),
            pt.nn.Linear(Nd, Nd),
            pt.nn.ELU(),
            pt.nn.Linear(Nd, Nd),
        )
        self.qrq = pt.nn.Sequential(
            pt.nn.Linear(Nd, Nd),
            pt.nn.ELU(),
            pt.nn.Linear(Nd, Nd),
            pt.nn.ELU(),
            pt.nn.Linear(Nd, Nd),
        )
        self.prk = pt.nn.Sequential(
            pt.nn.Linear(Nd, Nd),
            pt.nn.ELU(),
            pt.nn.Linear(Nd, Nd),
            pt.nn.ELU(),
            pt.nn.Linear(Nd, Nd),
        )
        self.prv = pt.nn.Sequential(
            pt.nn.Linear(Nd, Nd),
            pt.nn.ELU(),
            pt.nn.Linear(Nd, Nd),
            pt.nn.ELU(),
            pt.nn.Linear(Nd, Nd),
        )
        self.prq = pt.nn.Sequential(
            pt.nn.Linear(Nd, Nd),
            pt.nn.ELU(),
            pt.nn.Linear(Nd, Nd),
            pt.nn.ELU(),
            pt.nn.Linear(Nd, Nd),
        )

    def forward(self, X, ids_topk, q0, M):
        q = self.em.forward(q0)
        p0 = pt.zeros((q.shape[0]+1, X.shape[1], q.shape[1]), device=X.device)
        q, ids_topk, D_nn, R_nn = unpack_state_features(X, ids_topk, q)
        qa, pa, _, _, _ = self.sum.forward((q, p0, ids_topk, D_nn, R_nn))
        qr, pr = self.spl.forward(qa[1:], pa[1:], M)
        qr_k = self.qrk.forward(qr.unsqueeze(1))
        qr_v = self.qrv.forward(qr.unsqueeze(1))
        qr_q = self.qrq.forward(qr.unsqueeze(1))
        pr_norm = pt.norm(pr, dim=1)
        pr_k = self.prk.forward(pr_norm.unsqueeze(1))
        pr_v = self.prv.forward(pr_norm.unsqueeze(1))
        pr_q = self.prq.forward(pr_norm.unsqueeze(1))
        dk = pt.nn.Parameter(pt.sqrt(pt.tensor(1).float()), requires_grad=False)
        Mpr = pt.nn.functional.softmax(pt.matmul(qr_q, pr_k.transpose(1,2)) / dk, dim=2)
        Mqr = pt.nn.functional.softmax(pt.matmul(pr_q, qr_k.transpose(1,2)) / dk, dim=2)
        qr_ca = pt.matmul(Mpr, pr_v).squeeze(1)
        pr_ca = pt.matmul(Mqr, qr_v).squeeze(1)
        zr = pt.cat([qr_ca, pr_ca], dim=1)
        zr = zr.unsqueeze(1)
        zr = self.transformer_encoder.forward(zr)
        zr = zr.squeeze(1)
        z = self.dm.forward(zr)
        return z
