import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import csv

def train_model(data_loader, model,device,optimizer,x_dim,model_type):
    model.train()
    overall_loss = 0
    rep_loss = 0
    m_loss = 0
    v_loss = 0
    for batch_idx, (input, output) in enumerate(data_loader):
        #x = x.view(batch_size, x_dim)
        input = input.to(device)
        output = output.float()
        output = output.to(device)

        optimizer.zero_grad()

        pred, mean, log_var = model(input)

        loss,reproduction_loss, var_loss, mean_loss = loss_function(input.view(-1,x_dim), pred.view(-1,x_dim), mean, log_var,model_type)
        
        overall_loss += loss.item()
        rep_loss += reproduction_loss.item()
        m_loss += mean_loss.item()
        v_loss += var_loss.item()
        loss.backward()
        optimizer.step()
    return overall_loss / (batch_idx+1), rep_loss/(batch_idx+1), v_loss/(batch_idx+1), m_loss/(batch_idx+1)
            
def test_model(data_loader, model,device,x_dim,model_type):
    model.eval()  
    overall_loss = 0
    rep_loss = 0
    m_loss = 0
    v_loss = 0
    for batch_idx, (input, output) in enumerate(data_loader):
        input = input.to(device)
        output = output.float()
        output = output.to(device)
        pred, mean, log_var = model(input)
        loss,reproduction_loss, var_loss, mean_loss = loss_function(input.view(-1,x_dim), pred.view(-1,x_dim), mean, log_var,model_type)
        overall_loss += loss.item()
        rep_loss += reproduction_loss.item()
        m_loss += mean_loss.item()
        v_loss += var_loss.item()
    return overall_loss / (batch_idx+1), rep_loss/(batch_idx+1), v_loss/(batch_idx+1), m_loss/(batch_idx+1)
            


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
    # print("x_hat max: {}, x_hat min: {}".format(torch.max(x_hat), torch.min(x_hat) ))
    # print("mean max: {}, mean min: {}".format(torch.max(mean), torch.min(mean) ))
    # print("mean_loss: {}, var_loss: {}".format(mean_loss,var_loss))
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

def plot_ternary(simplex_points,loaded_model,im_x, im_y,latent_dim, output_file):
    #simplex_points = mean[:3,:]
    n_side = 10
    delta_n = 1/(n_side-1)
    delta_coord = 30
    n_total = int((n_side+1)*n_side/2)
    t_array = torch.zeros((n_total,3))
    coord_array = torch.zeros((n_total,2),dtype=torch.int32)
    i_idx = 0
    for i_row in range(n_side):
        for i_point in range(n_side-i_row):
            begin_coord = int((i_row+1)*delta_coord/2)
            t_array[i_idx,:] = torch.tensor([1- delta_n*i_point-delta_n*i_row,delta_n*i_point,delta_n*i_row])
            coord_array[i_idx,:] = torch.tensor([begin_coord+i_point*delta_coord,i_row*delta_coord])
            i_idx += 1
    
    selected_points = simplex_points[:,:,0,0]
    print("selected_ponts shape: {}, t_array shape: {}".format(selected_points.shape,t_array.shape ))
    all_points = torch.einsum('ik,kj->ij',t_array,selected_points)

    all_points = torch.tile(all_points[:,:,None,None],(1,1,10,6))

    real_all_points = all_points[:,:latent_dim//2,:,:]
    image_all_points = all_points[:,latent_dim//2:,:,:]
    all_points = torch.complex(real_all_points, image_all_points)

    interpolate_list = loaded_model.Decoder(all_points)
    fig, ax = plt.subplots(1,1)
    ax.set_xlim(0, int((n_side+1)*delta_coord))
    ax.set_ylim(0, int((n_side+1)*delta_coord))
    coordinatesList = [[0, 0], [100, 200], [200, 200]]
    cmap = 'Greens'
    cmap = plt.get_cmap(cmap) 
    for idx in range(n_total):
        tx, ty = coord_array[idx,0],coord_array[idx,1]
        ax.imshow(interpolate_list[idx].view(im_x, im_y).cpu().detach().numpy(),cmap=cmap, vmin=0, vmax=1.0,extent=(tx, tx + 28, ty, ty + 28))
    ax.axis("off")   
    fig.savefig(output_file,dpi = 450)