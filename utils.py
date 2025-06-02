import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import h5py


def show_image(x):
        fig = plt.figure()
        plt.imshow(x.cpu().numpy())
        plt.show()

def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
    return reproduction_loss + KLD
    
    
def load_mat(filename):
### write a load_mat code here (you can ask an LLM)