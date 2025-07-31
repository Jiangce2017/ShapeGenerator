import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import os.path as osp
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Not always needed, but sometimes required for 3D


from sklearn.model_selection import train_test_split

from models_3d import Model
from dataset import ABCDataset
from utils import loss_function, load_mat, show_image, CombinedDataset,Logger,plot_ternary

if __name__ == '__main__':
    print("Program started")
    cuda = True
    device = torch.device("cuda" if cuda else "cpu")
    train_resolution = 20
    im_x = train_resolution
    im_y = train_resolution
    im_z = train_resolution
    modes1 = 10
    modes2 = 6
    modes3 = 6
    batch_size = 64
    output_dim  = 1
    hidden_dim = 64
    latent_dim = 64
    lr = 1e-3
    epochs = 500

    model_type = 'Freq_FNO3D'
    dataset_root_dir =  '/scratch/jc14407/datasets'

    data_name = 'abc_voxelized_20'
    exp_name = model_type + '_' + data_name

    dataset_path = osp.join(dataset_root_dir, data_name)
    if not osp.exists(dataset_path):
        print("Dataset not found at", dataset_path)
        exit(1)  
    
    checkpoints_dir = './checkpoints'
    if not osp.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    results_dir = './results'
    if not osp.exists(results_dir):
        os.makedirs(results_dir)
    
    model_file = osp.join(checkpoints_dir, exp_name+"_.pth")
    if osp.exists(model_file):
        print("Loading model from", model_file)
        model = torch.load(model_file, map_location=device)
    else:
        print("No pre-trained model found, creating a new one")
        # Create a new model instance
        model = Model(output_dim, hidden_dim, latent_dim,device,model_type,im_x,im_y,im_z,modes1,modes2,modes3).to(device)

    model.device = device

    dataset = ABCDataset(dataset_path, voxelized = True)
    print("dataset loaded, size:", len(dataset))

    train_dataset, test_dataset = train_test_split(dataset, test_size=0.1, random_state=42)
    kwargs = {'num_workers': 1, 'pin_memory': False} 
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,drop_last=True, **kwargs)
    test_loader  = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False,drop_last=False, **kwargs)

    model.eval()
    with torch.no_grad():
        it = iter(train_loader)
        batch = next(it)
        input_voxels = batch.to(device)
        pred, mean, log_var = model(input_voxels)

        batch_size = input_voxels.size(0)
        loss, reproduction_loss, var_loss, mean_loss = loss_function(
            input_voxels.view(batch_size, -1),  # (B, x_dim^3)
            pred.reshape(batch_size, -1),
            mean,
            log_var,
            model_type
        )
    print("Loss: {}, Reproduction Loss: {}, Variance Loss: {}, Mean Loss: {}".format(
        loss.item(), reproduction_loss.item(), var_loss.item(), mean_loss.item()
    ))
    #pred = torch.sigmoid(pred)
    pred[pred < 0.5] = 0
    pred[pred >= 0.5] = 1
    x_hat_list = []
    x_hat_title = []
    x_hat_list.append(input_voxels[0].cpu().detach().numpy().reshape(im_x,im_y,im_z))
    x_hat_title.append("Input")
    x_hat_list.append(pred[0].cpu().detach().numpy().reshape(im_x,im_y,im_z))
    x_hat_title.append("Predicted")


    latent_vector = mean[[0]]

    real_latent_vector = latent_vector[:,:latent_dim//2,:,:,:]
    image_latent_vector = latent_vector[:,latent_dim//2:,:,:,:]
    latent_vector = torch.complex(real_latent_vector, image_latent_vector)

    latent_vector = torch.tile(latent_vector,(1,1,modes1,modes2,modes3))
    print("latent_vector shape:{}".format(latent_vector.shape))
    x_hat = model.Decoder(latent_vector,20,20,20)
    #show_image(x_hat.cpu().detach().numpy().reshape(30,30,30))
    x_hat[x_hat < 0.5] = 0
    x_hat[x_hat >= 0.5] = 1
    x_hat_list.append(x_hat.cpu().detach().numpy().reshape(20,20,20))
    x_hat_title.append("Reconstructed")

    fig = plt.figure(figsize=(15, 5))
    # First subplot
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax1.voxels(x_hat_list[0], edgecolor='k')
    ax1.set_title(x_hat_title[0])
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_aspect('equal') 

    # Second subplot
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax2.voxels(x_hat_list[1], edgecolor='k')
    ax2.set_title(x_hat_title[1])
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_aspect('equal')         

    # Third subplot         
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax3.voxels(x_hat_list[2], edgecolor='k')
    ax3.set_title(x_hat_title[2])
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z') 
    ax3.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(osp.join(results_dir, exp_name + '_reconstruction.png'))



