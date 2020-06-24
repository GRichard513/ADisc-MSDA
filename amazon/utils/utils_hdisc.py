import numpy as np
import torch
import pandas as pd

def split_source_target(X_amazon, y_amazon, target, device='cuda:0', merge=False):
    sources = np.where(np.arange(len(X_amazon))!=target)[0]
    X_t = X_amazon[target]
    y_t = y_amazon[target] .squeeze()
    if merge:
        X_s = [np.concatenate([X_amazon[s] for s in sources])]
        y_s = [np.concatenate([y_amazon[s] for s in sources])]
        
    else:
        X_s = np.array([X_amazon[s] for s in sources])
        y_s = np.array([y_amazon[s] for s in sources])
    X_s = [torch.Tensor(x).to(device) for x in X_s]
    X_t = torch.Tensor(X_t).to(device)
    y_s = [torch.Tensor(y).to(device).unsqueeze(1) for y in y_s]
    y_t = torch.Tensor(y_t).to(device).unsqueeze(1)
    return X_s, X_t, y_s, y_t

def batch_loader(X_s, y_s, batch_size = 64, shuffle = True, random_state=0):
    if random_state is not None:
        np.random.seed(random_state)
    inputs, targets = [x.clone() for x in X_s], [y.clone() for y in y_s]
    input_sizes = [X_s[i].shape[0] for i in range(len(X_s))]
    max_input_size = max(input_sizes)
    n_sources = len(X_s)
    if shuffle:
        for i in range(n_sources):
            r_order = np.arange(input_sizes[i])
            np.random.shuffle(r_order)
            inputs[i], targets[i] = inputs[i][r_order, :], targets[i][r_order]
    num_blocks = int(max_input_size / batch_size)
    for j in range(num_blocks):
        xs, ys = [], []
        for i in range(n_sources):
            ridx = np.random.choice(input_sizes[i], batch_size)
            xs.append(inputs[i][ridx, :])
            ys.append(targets[i][ridx])
        yield xs, ys
        
def val_split(X_s, y_s, frac_train = 0.9):
    n_domains = len(X_s)
    idx = [np.random.permutation(np.arange(len(x))) for x in X_s]
    idx_train = [idx[i][:int(len(X_s[i])*frac_train)] for i in range(n_domains)]
    idx_val = [idx[i][int(len(X_s[i])*frac_train):] for i in range(n_domains)]
    X_train, y_train = [X_s[i][idx_train[i]]  for i in range(n_domains)], [y_s[i][idx_train[i]]  for i in range(n_domains)]
    X_val, y_val = [X_s[i][idx_val[i]]  for i in range(n_domains)], [y_s[i][idx_val[i]]  for i in range(n_domains)]
    return X_train, X_val, y_train, y_val

def save_result(result, filepath, domain_list):
    median = {}
    for k in domain_list:
        median[k] = []
    for r in range(len(result)):
        for k in domain_list:
            median[k].append(result[r][k])
    df = pd.DataFrame(median)
    df.to_csv(filepath)
    