import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os.path as osp
import os
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import Adam

from sklearn.model_selection import train_test_split

from models_3d import Model
from utils import loss_function, load_mat, show_image, CombinedDataset,Logger,train_3D_model, test_3D_model
from dataset import ABCDataset

if __name__ == '__main__':
    print("Program started")
    cuda = True
    device = torch.device("cuda" if cuda else "cpu")
    #train_model = True
    train_resolution = 40
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
    dataset_root_dir = '/scratch/jc14407/datasets'

    data_name = 'abc_voxelized_40'
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


    ## setup logger
    train_logger = Logger(
        osp.join(results_dir, exp_name+'_train.log'),
        ['ep', 'train_loss','train_rep','train_var','train_mean']
    )
    test_logger = Logger(
        osp.join(results_dir, exp_name+'_test.log'),
        ['ep', 'test_loss','test_rep','test_var','test_mean']
    )
    kwargs = {'num_workers': 1, 'pin_memory': False} 

    dataset = ABCDataset(dataset_path, voxelized = True)
    print("dataset loaded, size:", len(dataset))

    train_dataset, test_dataset = train_test_split(dataset, test_size=0.1, random_state=42)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,drop_last=True, **kwargs)
    test_loader  = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False,drop_last=False, **kwargs)

    print("Start training...")
    optimizer = Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        overall_loss, rep_loss, v_loss, m_loss= train_3D_model(train_loader,model,device,optimizer,train_resolution,model_type)
        print("\tEpoch", epoch + 1, "complete!", "\tAverage Train Loss: ", overall_loss)
        train_logger.log({
        'ep': epoch,             
        'train_loss': overall_loss,
        'train_rep': rep_loss,
        'train_var': v_loss, 
        'train_mean': m_loss
        })

        torch.save(model, model_file)
        if epoch % 10 == 0:
            overall_loss, rep_loss, v_loss, m_loss = test_3D_model(test_loader, model,device,train_resolution,model_type)
            print("\tEpoch", epoch + 1, "complete!", "\tAverage Test Loss: ", overall_loss)
            test_logger.log({
            'ep': epoch,             
            'test_loss': overall_loss,
            'test_rep': rep_loss,
            'test_var': v_loss,
            'test_mean': m_loss
            })
    print("Finish!!")
    torch.save(model, model_file)