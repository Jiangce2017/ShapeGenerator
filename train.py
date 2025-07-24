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

from models import Model
from utils import loss_function, load_mat, show_image, CombinedDataset,Logger,train_model, test_model

if __name__ == '__main__':
    print("Program started")
    cuda = True
    device = torch.device("cuda" if cuda else "cpu")
    #train_model = True
    im_x = 50
    im_y = 50 
    modes1 = 10
    modes2 = 6
    model_type = 'Freq_FNO'
    dataset_root_dir = '/scratch/jc14407/datasets'

    dataset_path = osp.join(dataset_root_dir, 'Wang/ShapeSpace.mat') 
    if not osp.exists(dataset_path):
        print("Dataset not found at", dataset_path)
        exit(1)  
    if not osp.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not osp.exists('results'):
        os.makedirs('results')
    results_dir = './results'
    model_file = osp.join("./checkpoints",model_type+"_model.pth")
    batch_size = 64
    x_dim  = 2500
    hidden_dim = 64
    latent_dim = 64
    lr = 1e-3
    epochs = 1000

    ## setup logger

    train_logger = Logger(
        osp.join(results_dir, model_type+'_train.log'),
        ['ep', 'train_loss','train_rep','train_var','train_mean']
    )
    test_logger = Logger(
        osp.join(results_dir, model_type+'_test.log'),
        ['ep', 'test_loss','test_rep','test_var','test_mean']
    )
    kwargs = {'num_workers': 1, 'pin_memory': False} 

    mat_data = load_mat(dataset_path)
    dataset = mat_data['ShapeSpace']
    dataset = dataset.astype(np.float32)
    input_dataset = dataset
    print("dataset loaded")

    train_dataset, test_dataset = train_test_split(input_dataset, test_size=0.1, random_state=42)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,drop_last=True, **kwargs)
    test_loader  = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False,drop_last=False, **kwargs)
        
    model = Model(x_dim, hidden_dim, latent_dim,device,model_type,im_x,im_y,modes1,modes2).to(device)

    print("Start training...")
    optimizer = Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        overall_loss, rep_loss, m_loss, v_loss = train_model(train_loader,model,device,optimizer,x_dim,model_type)
        print("\tEpoch", epoch + 1, "complete!", "\tAverage Train Loss: ", overall_loss)
        train_logger.log({
        'ep': epoch,             
        'train_loss': overall_loss,
        'train_rep': rep_loss,
        'train_var': v_loss, 
        'train_mean': m_loss
        })


        if epoch % 10 == 0:
            torch.save(model, model_file)
            overall_loss, rep_loss, m_loss, v_loss = test_model(test_loader, model,device,x_dim,model_type)
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