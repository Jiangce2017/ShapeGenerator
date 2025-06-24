import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import h5py

def plot_metric(values, title, ylabel, filename):
        plt.figure()
        plt.plot(values)
        plt.title(title)
        plt.xlabel("Batch")
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.savefig(f"plots/plots4/{filename}.png")
        plt.close()

def show_image(x):
        fig = plt.figure()
        plt.imshow(x.cpu().numpy())
        plt.show()

# def loss_function(x, x_hat, mean, log_var):
#     reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
#     Klog_var = torch.clamp(log_var, min=-10, max=10)  # double protection
#     KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
#     KLD = torch.nan_to_num(KLD, nan=0.0, posinf=1e5, neginf=-1e5)
#     return reproduction_loss + KLD
    
    
def load_mat(filename):
    with h5py.File(filename, 'r') as f:
        data = {}
        for k, v in f.items():
            data[k] = v[:]  # Load data into memory
    return data

def loss_function(x, x_hat, mean, log_var, beta = 1.0):
    recon_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return recon_loss + beta * kld

