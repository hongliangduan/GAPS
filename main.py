import os
import json
import numpy as np
import torch as pt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from src.logger import Logger
from src.scoring import bc_scoring, bc_score_names, nanmean
from src.config import config_data, config_model, config_runtime
from src.data_handler import Dataset, collate_batch_data
from src.dataset import sele_sid, sele_max_ba, sele_interface_types
from model.model import Model

def eval(model, device, batch_data, criterion, pos_ratios, pos_weight_factor, global_step):
    X, ids_topk, q, M, y = [data.to(device) for data in batch_data]
    z = model.forward(X, ids_topk, q, M)
    pos_ratios += (pt.mean(y,dim=0).detach() - pos_ratios) / (1.0 + np.sqrt(global_step))
    criterion.pos_weight = pos_weight_factor * (1.0 - pos_ratios) / (pos_ratios + 1e-6)
    dloss = criterion(z, y)
    loss_factors = (pos_ratios / pt.sum(pos_ratios)).reshape(1,-1)
    losses = (loss_factors * dloss) / dloss.shape[0]
    return losses, y.detach(), pt.sigmoid(z).detach()

def log(logger, writer, scores, global_step, pos_ratios, step_type):
    pr_str = ', '.join([f"{r:.4f}" for r in pos_ratios])
    logger.print(f"{step_type}> [{global_step}] loss={scores['loss']:.4f}, pos_ratios=[{pr_str}]")
    summary_stats = {k:scores[k] for k in scores if not np.isnan(scores[k])}
    summary_stats['global_step'] = int(global_step)
    summary_stats['pos_ratios'] = list(pos_ratios.cpu().numpy())
    summary_stats['step_type'] = step_type
    logger.store(**summary_stats)
    for key in scores:
        writer.add_scalar(step_type+'/'+key, scores[key], global_step)
    for c in np.unique([key.split('/')[0] for key in scores if len(key.split('/')) == 2]):
        logger.print(f'[{c}] loss={scores[c+"/loss"]:.3f}, ' + ', '.join([f'{sn}={scores[c+"/"+sn]:.3f}' for sn in bc_score_names]))

def setup_data(config, sids_selection_filepath, dataset_filepath):
    sids_sel = np.genfromtxt(sids_selection_filepath, dtype=np.dtype('U'))
    dataset = Dataset(dataset_filepath)
    m = sele_sid(dataset, sids_sel)
    m &= sele_max_ba(dataset, config['max_ba'])
    m &= (dataset.sizes[:,0] <= config['max_size'])
    m &= (dataset.sizes[:,1] >= config['min_num_res'])
    m &= sele_interface_types(dataset, config['l_types'], np.concatenate(config['r_types']))
    dataset.update_mask(m)
    dataset.set_types(config['l_types'], config['r_types'])
    dataloader = pt.utils.data.DataLoader(dataset, batch_size=config_runtime['batch_size'], shuffle=True, num_workers=8, collate_fn=collate_batch_data, pin_memory=True, prefetch_factor=2)
    return dataloader

def score(eval_results):
    sum_losses, scores = [], []
    for losses, y, p in eval_results:
        sum_losses.append(pt.sum(losses, dim=0))
        scores.append(bc_scoring(y, p))
    m_losses = pt.mean(pt.stack(sum_losses, dim=0), dim=0).numpy()
    m_scores = nanmean(pt.stack(scores, dim=0)).numpy()
    scores = {'loss': float(np.sum(m_losses))}
    for i in range(m_losses.shape[0]):
        scores[f'{i}/loss'] = m_losses[i]
        for j in range(m_scores.shape[0]):
            scores[f'{i}/{bc_score_names[j]}'] = m_scores[j,i]
    return scores

def train(config_data, config_model, config_runtime, output_path, finetune=False):
    logger = Logger(output_path, 'train')
    logger.print(">>> Configuration")
    logger.print(config_data)
    logger.print(config_runtime)
    device = pt.device(config_runtime['device'])
    model = Model(config_model)
    if finetune == True:
        model.load_state_dict(pt.load('checkpoints/model.pt'))
        dataset_filepath = 'data/pepnn.h5'
        train_selection_filepath = 'util/pepnn_train.txt'
        val_selection_filepath = 'util/pepnn_val.txt'
    elif finetune == False:
        dataset_filepath = 'data/scannet.h5'
        train_selection_filepath = 'util/scannet_train.txt'
        val_selection_filepath = 'util/scannet_val.txt'
    logger.print(">>> Model")
    logger.print(model)
    logger.print(f"> {sum([int(pt.prod(pt.tensor(p.shape))) for p in model.parameters()])} parameters")
    model_filepath = os.path.join(output_path, 'model_ckpt.pt')
    if os.path.isfile(model_filepath) and config_runtime["reload"]:
        logger.print("Reloading model from save file")
        model.load_state_dict(pt.load(model_filepath))
        global_step = json.loads([l for l in open(logger.log_lst_filepath, 'r')][-1])['global_step']
        pos_ratios = pt.from_numpy(np.array(json.loads([l for l in open(logger.log_lst_filepath, 'r')][-1])['pos_ratios'])).float().to(device)
    else:
        global_step = 0
        pos_ratios = 0.5*pt.ones(len(config_data['r_types']), dtype=pt.float).to(device)
    logger.print(">>> Loading data")
    dataloader_train = setup_data(config_data, train_selection_filepath, dataset_filepath)
    dataloader_val = setup_data(config_data, val_selection_filepath, dataset_filepath)
    logger.print(f"> training data size: {len(dataloader_train)}")
    logger.print(f"> validation data size: {len(dataloader_val)}")
    logger.print(">>> Starting training")
    model = model.to(device)
    criterion = pt.nn.BCEWithLogitsLoss(reduction="none")
    optimizer = pt.optim.Adam(model.parameters(), lr=config_runtime["learning_rate"])
    logger.restart_timer()
    writer = SummaryWriter(os.path.join(output_path, 'tb'))
    min_loss = 1e9
    batch_data = collate_batch_data([dataloader_train.dataset.get_largest()])
    optimizer.zero_grad()
    losses, _, _ = eval(model, device, batch_data, criterion, pos_ratios, config_runtime['pos_weight_factor'], global_step)
    loss = pt.sum(losses)
    loss.backward()
    optimizer.step()
    for _ in range(config_runtime['num_epochs']):
        model = model.train()
        train_results = []
        for batch_train_data in tqdm(dataloader_train):
            global_step += 1
            optimizer.zero_grad()
            losses, y, p = eval(model, device, batch_train_data, criterion, pos_ratios, config_runtime['pos_weight_factor'], global_step)
            loss = pt.sum(losses)
            loss.backward()
            optimizer.step()
            train_results.append([losses.detach().cpu(), y.cpu(), p.cpu()])
            if (global_step+1) % config_runtime["log_step"] == 0:
                with pt.no_grad():
                    scores = score(train_results, device=device)
                    train_results = []
                    log(logger, writer, scores, global_step, pos_ratios, "train")
                    model_filepath = os.path.join(output_path, 'model_ckpt.pt')
                    pt.save(model.state_dict(), model_filepath)
            if (global_step+1) % config_runtime["eval_step"] == 0:
                model = model.eval()
                with pt.no_grad():
                    test_results = []
                    for step_te, batch_test_data in enumerate(dataloader_val):
                        losses, y, p = eval(model, device, batch_test_data, criterion, pos_ratios, config_runtime['pos_weight_factor'], global_step)
                        test_results.append([losses.detach().cpu(), y.cpu(), p.cpu()])
                        if step_te >= config_runtime['eval_size']:
                            break
                    scores = score(test_results, device=device)
                    log(logger, writer, scores, global_step, pos_ratios, "test")
                    if min_loss >= scores['loss']:
                        min_loss = scores['loss']
                        model_filepath = os.path.join(output_path, 'model.pt')
                        logger.print("> saving model at {}".format(model_filepath))
                        pt.save(model.state_dict(), model_filepath)
                model = model.train()

if __name__ == '__main__':
    train(config_data, config_model, config_runtime, 'checkpoints_tmp', finetune=True)