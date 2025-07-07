import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import csv

class Logger(object):
    def __init__(self, path, header):
        self.log_file = open(path, 'a')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()

class CombinedDataset(Dataset):
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data
    
    def __len__(self):
        return min(len(self.input_data), len(self.output_data))
    
    def __getitem__(self, idx):
        return self.input_data[idx], self.output_data[idx]

def show_image(x):
        fig = plt.figure()
        cmap = 'Greens'
        plt.imshow(x,cmap=cmap)
        plt.show()

def loss_function_NO(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat,x, reduction='mean')
    #reproduction_loss =  F.mse_loss(x, x_hat, reduction='mean') 
    var_loss = torch.mean(torch.exp(log_var))
    mean_loss = 1/(1+torch.exp(-16*(torch.max(mean.pow(2))-1)))
    KLD = - 0.5 * torch.mean(1+ log_var - mean.pow(2) - log_var.exp())
    print("x_hat max: {}, x_hat min: {}".format(torch.max(x_hat), torch.min(x_hat) ))
    print("mean max: {}, mean min: {}".format(torch.max(mean), torch.min(mean) ))
    print("mean_loss: {}, var_loss: {}".format(mean_loss,var_loss))
    
    total_loss = reproduction_loss + var_loss + mean_loss
    return total_loss, reproduction_loss, var_loss, torch.max(torch.abs(mean))

def loss_function(x, x_hat, mean, log_var,model_type):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat,x, reduction='mean')
    KLD = - 0.5 * torch.mean(1+ log_var - mean.pow(2) - log_var.exp())
    var_loss = torch.mean(torch.exp(log_var))
    mean_loss = 1/(1+torch.exp(-16*(torch.max(mean.pow(2))-1)))
    print("x_hat max: {}, x_hat min: {}".format(torch.max(x_hat), torch.min(x_hat) ))
    print("mean max: {}, mean min: {}".format(torch.max(mean), torch.min(mean) ))
    print("mean_loss: {}, var_loss: {}".format(mean_loss,var_loss))
    if model_type == 'FNO' or model_type == 'Freq_FNO':
        total_loss = reproduction_loss + var_loss + mean_loss
    else:
        total_loss = reproduction_loss + KLD
    return total_loss, reproduction_loss, var_loss, torch.max(torch.abs(mean))
    
def load_mat(filename):
    with h5py.File(filename, 'r') as f:
        data = {}
        for k, v in f.items():
            data[k] = v[:]  # Load data into memory
    return data

# normalization, pointwise gaussian
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001, time_last=True):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T in 1D
        # x could be in shape of ntrain*w*l or ntrain*T*w*l or ntrain*w*l*T in 2D
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps
        self.time_last = time_last # if the time dimension is the last dim

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        # sample_idx is the spatial sampling mask
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if self.mean.ndim == sample_idx.ndim or self.time_last:
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if self.mean.ndim > sample_idx.ndim and not self.time_last:
                    std = self.std[...,sample_idx] + self.eps # T*batch*n
                    mean = self.mean[...,sample_idx]
        # x is in shape of batch*(spatial discretization size) or T*batch*(spatial discretization size)
        x = (x * std) + mean
        return x

    def to(self, device):
        if torch.is_tensor(self.mean):
            self.mean = self.mean.to(device)
            self.std = self.std.to(device)
        else:
            self.mean = torch.from_numpy(self.mean).to(device)
            self.std = torch.from_numpy(self.std).to(device)
        return self

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()