from datetime import datetime
from .data_encoding import categ_to_resnames

config_data = {
    'max_ba': 1,
    'max_size': 1024*8,
    'min_num_res': 48,
    'l_types': categ_to_resnames['protein'],
    'r_types': [categ_to_resnames['protein']],
}

config_model = {
    "em": {'N0': 30, 'N1': 32},
    "sum": [
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 8},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 8},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 8},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 8},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 8},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 8},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 16},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 16},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 16},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 16},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 16},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 16},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 32},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 32},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 32},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 32},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 32},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 32},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 64},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 64},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 64},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 64},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 64},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 64},
    ],
    "spl": {'N0': 32, 'N1': 32, 'Nh': 4},
    "dm": {'N0': 32, 'N1': 32, 'N2': 1},
    "transformer_encoder": {'head': 8, 'layer': 8},
}

tag = datetime.now().strftime("_%Y-%m-%d_%H-%M")

config_runtime = {
    'run_name': 'i_v4_1'+tag,
    'output_dir': 'save',
    'reload': True,
    'device': 'cuda:1',
    'num_epochs': 100,
    'batch_size': 1,
    'log_step': 1024,
    'eval_step': 2048,
    'eval_size': 1024,
    'learning_rate': 1e-5,
    'pos_weight_factor': 0.5,
}
