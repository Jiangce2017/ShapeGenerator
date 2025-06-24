import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import h5py


def show_image(x):
        fig = plt.figure()
        cmap = 'Greens'
        plt.imshow(x.cpu().numpy(),cmap=cmap)
        plt.show()

def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
    return reproduction_loss + KLD
    
    
def load_mat(filename):
    with h5py.File(filename, 'r') as f:
        data = {}
        for k, v in f.items():
            data[k] = v[:]  # Load data into memory
    return data