import numpy as np
import torch as pt
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

bc_score_names = ['acc','ppv','npv','tpr','tnr','mcc','auc','std']

@pt.jit.script
def binary_classification(y, q):
    TP = pt.sum(q * y, dim=0)
    TN = pt.sum((1.0-q) * (1.0-y), dim=0)
    FP = pt.sum(q * (1.0-y), dim=0)
    FN = pt.sum((1.0-q) * y, dim=0)
    P = pt.sum(y, dim=0)
    N = pt.sum(1.0-y, dim=0)
    return TP, TN, FP, FN, P, N

@pt.jit.script
def acc(TP, TN, FP, FN):
    return (TP + TN) / (TP + TN + FP + FN)

@pt.jit.script
def ppv(TP, FP, P):
    v = TP / (TP + FP)
    v[~(P>0)] = np.nan 
    return v

@pt.jit.script
def npv(TN, FN, N):
    v = TN / (TN + FN)
    v[~(N>0)] = np.nan 
    return v

@pt.jit.script
def tpr(TP, FN):
    v = TP / (TP + FN)
    v[pt.isinf(v)] = np.nan
    return v

@pt.jit.script
def tnr(TN, FP):
    v = TN / (TN + FP)
    v[pt.isinf(v)] = np.nan
    return v

@pt.jit.script
def mcc(TP, TN, FP, FN):
    v = ((TP*TN) - (FP*FN)) / pt.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    v[pt.isinf(v)] = np.nan
    return v

@pt.jit.script
def recall(TP, FN):
    recall = TP / (TP + FN)
    return recall

@pt.jit.script
def precision(TP, FP):
    precision = TP / (TP + FP)
    return precision

@pt.jit.script
def f1(TP, FN, FP):
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    f1 = 2*(recall*precision)/(recall+precision)
    f1[pt.isnan(f1)] = 0
    f1[pt.isinf(f1)] = 0
    return f1

def roc_auc(y, p, P, N):
    m = (P > 0) & (N > 0)
    v = pt.zeros(y.shape[1], dtype=pt.float32, device=y.device)
    if pt.any(m):
        a = np.array(roc_auc_score(y[:,m].cpu().numpy(), p[:,m].cpu().numpy(), average=None))
        v[m] = pt.from_numpy(a).float().to(y.device)
    v[~m] = np.nan
    return v

@pt.jit.script
def nanmean(x):
    return pt.nansum(x, dim=0) / pt.sum(~pt.isnan(x), dim=0)

def bc_scoring(y, p):
    q = pt.round(p)
    TP, TN, FP, FN, P, N = binary_classification(y, q)
    scores = pt.stack([
        acc(TP, TN, FP, FN),
        ppv(TP, FP, P),
        npv(TN, FN, N),
        tpr(TP, FN),
        tnr(TN, FP),
        mcc(TP, TN, FP, FN),
        roc_auc(y, p, P, N),
        pt.std(p, dim=0),
    ])
    return scores

def reg_scoring(y, p):
    return {
        'mse': float(pt.mean(pt.square(y - p)).cpu().numpy()),
        'mae': float(pt.mean(pt.abs(y - p)).cpu().numpy()),
        'rmse': float(pt.sqrt(pt.mean(pt.square(y - p))).cpu().numpy()),
        'pcc': pearsonr(y.cpu().numpy(), p.cpu().numpy())[0] if not pt.allclose(y,y[0]) else np.nan,
        'std': float(pt.std(p).cpu().numpy()),
    }

def bc_scoring_plus(y, p):
    q = pt.round(p)
    TP, TN, FP, FN, P, N = binary_classification(y, q)
    scores = pt.stack([
        recall(TP, FN),
        precision(TP, FP),
        f1(TP, FP, FN),
    ])
    return scores