import os
import pickle
import random
from itertools import chain

from einops import rearrange
import numpy as np
import torch
from scipy.io import loadmat


def generate_indices(input_step, pred_step, t_length, block_size=None):
    if block_size is None:
        block_size = t_length
    offsets_in_block = np.arange(input_step, block_size - pred_step + 1)
    assert t_length % block_size == 0, "t_length % block_size != 0"
    random_t_list = []
    for block_start in range(0, t_length, block_size):
        random_t_list += (offsets_in_block + block_start).tolist()
    np.random.shuffle(random_t_list)
    return random_t_list

def batch_generater(data, bs, n_nodes, input_step, pred_step, block_size=None):
    t, n, d = data.shape
    random_t_list = generate_indices(input_step, pred_step, t_length=t, block_size=block_size)
    for batch_i in range(len(random_t_list) // bs):
        x = torch.zeros([bs, n_nodes, input_step, d])
        y = torch.zeros([bs, n_nodes, pred_step, d])
        t = torch.zeros([bs]).long()
        for data_i in range(bs):
            data_t = random_t_list.pop()
            x[data_i, :, :, :] = rearrange(data[data_t - input_step: data_t, :], "t n d -> n t d")
            y[data_i, :, :, :] = rearrange(data[data_t: data_t + pred_step, :], "t n d -> n t d")
            t[data_i] = data_t
        yield x, y, t

def prepross_data(data):
    T, N, D = data.shape
    new_data = np.zeros_like(data, dtype=float)
    for i in range(N):
        node = data[:, i, :]
        new_data[:, i, :] = (node - np.mean(node)) / np.std(node)
    return new_data

def data2batch(path,bs,n_nodes,input_step,pred_step,start,end,n_sub):
    file_path = os.path.join(path, 'trajs.pickle')
    graph_path_1 = os.path.join(path, 'GC_by_subject.npy')
    graph_path_2 = os.path.join(path, 'GC_bar.npy')
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    values = list(data.values())
    data = np.array(values)
    graph_sub = np.load(graph_path_1)
    graph_bar = np.load(graph_path_2)
    all_batch_data = []
    for i in range(n_sub):
        x = data[i, :, :]
        x = x[start:end, :]
        x = x[:, :, None]
        x = prepross_data(x)
        x = torch.tensor(x)
        batch_gen = batch_generater(x, bs, n_nodes, input_step, pred_step, block_size=None)
        batch_gen = list(batch_gen)
        all_batch_data.append(batch_gen)
    return all_batch_data, graph_sub, graph_bar

def split_list_ratio(data, val_ratio=0.1, seed=42):
    random.seed(seed)
    data = data.copy()
    random.shuffle(data)
    n_total = len(data)
    n_val = int(n_total * val_ratio)
    val_set = data[:n_val]
    train_set = data[n_val:]
    return train_set, val_set

def fmri2batch(path,n_sub,bs,input_step,pred_step):
    sim = loadmat(path)
    x = sim['ts']
    GC = sim['net']
    GC = GC.mean(axis=0)
    GC[GC != 0] = 1
    GC[GC == 0] = 0
    graph = GC
    all_batch_data = []
    for i in range(n_sub):
        start = i * 200
        end = start + 200
        x_sub = x[start:end, :]
        x_sub = x_sub[:, :, None]
        x_sub = prepross_data(x_sub)
        x_sub = torch.tensor(x_sub)
        n_nodes = x_sub.shape[1]
        batch_data = batch_generater(x_sub, bs, n_nodes, input_step, pred_step, block_size=None)
        batch_data = list(batch_data)
        all_batch_data.append(list(batch_data))
    train_data, val_data = split_list_ratio(all_batch_data, val_ratio=0.1)
    return list(chain.from_iterable(train_data)),list(chain.from_iterable(val_data)), graph