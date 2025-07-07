import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os.path as osp
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from models import Model
from utils import loss_function, load_mat, show_image, CombinedDataset,Logger

if __name__ == '__main__':

    """
        A simple implementation of Gaussian MLP Encoder and Decoder
    """
    cuda = False
    device = torch.device("cuda" if cuda else "cpu")
    train_model = False
    im_x = 50
    im_y = 50 
    modes1 = 10
    modes2 = 6
    model_type = 'Freq_FNO'
    dataset_path = './datasets/Wang/ShapeSpace.mat'
    results_dir = './results'
    model_file = osp.join("checkpoints",model_type+"_model.pth")
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
    kwargs = {'num_workers': 1, 'pin_memory': False} 

    mat_data = load_mat(dataset_path)
    dataset = mat_data['ShapeSpace']
    dataset = dataset.astype(np.float32)
    input_dataset = dataset[:512]


    output_dataset_path = './datasets/Wang/Physics.npy'
    with open(output_dataset_path, 'rb') as f:
        output_dataset = np.load(f)
        output_dataset = torch.from_numpy(output_dataset)
    combined_dataset = CombinedDataset(input_dataset, output_dataset)

    train_dataset, test_dataset = train_test_split(combined_dataset, test_size=0.1, random_state=42)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,drop_last=True, **kwargs)
    test_loader  = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False,drop_last=False, **kwargs)
        
    model = Model(x_dim, hidden_dim, latent_dim,device,model_type,im_x,im_y,modes1,modes2).to(device)

    if train_model:
        optimizer = Adam(model.parameters(), lr=lr)
        print("Start training VAE...")
        model.train()
        for epoch in range(epochs):
            overall_loss = 0
            rep_loss = 0
            m_loss = 0
            v_loss = 0
            for batch_idx, (input, output) in enumerate(train_loader):
                #x = x.view(batch_size, x_dim)



                input = input.to(device)
                output = output.float()
                output = output.to(device)

                optimizer.zero_grad()

                pred, mean, log_var = model(input)

                loss,reproduction_loss, var_loss, mean_loss = loss_function(input.view(batch_size,x_dim), pred.view(batch_size,x_dim), mean, log_var,model_type)
                
                overall_loss += loss.item()
                rep_loss += reproduction_loss.item()
                m_loss += mean_loss.item()
                v_loss += var_loss.item()
                loss.backward()
                optimizer.step()
                
            print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss / (batch_idx+1))
            train_logger.log({
            'ep': epoch,             
            'train_loss': overall_loss / (batch_idx+1),
            'train_rep': rep_loss/(batch_idx+1),
            'train_var': v_loss/(batch_idx+1),
            'train_mean': m_loss/(batch_idx+1)
            })
            torch.save(model, model_file)
            
        print("Finish!!")
        torch.save(model, model_file)
    
    else:
        model_file = osp.join("checkpoints","save_"+model_type+"_model.pth")
        loaded_model = torch.load(model_file)
        loaded_model.eval()
        with torch.no_grad():
            for batch_idx, (input, output) in enumerate(tqdm(train_loader)):

                input = input.to(device)
                
                pred, mean, log_var = loaded_model(input)
                loss, _, _, _ = loss_function(input.view(-1,x_dim), pred.view(-1,x_dim), mean, log_var,model_type)
                print("loss: {}".format(loss.item()))
                break
        
        pred = F.sigmoid(pred)
        show_image(input[0].cpu().detach().numpy().reshape(im_x,im_y))
        show_image(pred[0].cpu().detach().numpy().reshape(im_x,im_y))


        latent_vector = mean[[0]]

        real_latent_vector = latent_vector[:,:latent_dim//2,:,:]
        image_latent_vector = latent_vector[:,latent_dim//2:,:,:]
        latent_vector = torch.complex(real_latent_vector, image_latent_vector)

        latent_vector = torch.tile(latent_vector,(1,1,10,6))
        print("latent_vector shape:{}".format(latent_vector.shape))
        x_hat = loaded_model.Decoder(latent_vector,60,60)
        show_image(x_hat.cpu().detach().numpy().reshape(60,60))
        plt.close('all') 

        simplex_points = mean[:3,:]

        # radius = 3*dist_min
        # virtual_center = torch.mean(latent_points_arr,dim=0)
        # dist_2_virtual_center = L2_dist(latent_points_arr,virtual_center)
        # nearest_center_point = torch.max(dist_2_virtual_center,dim=0).indices
        # center = latent_points_arr[nearest_center_point,:]
        # simplex_points = torch.zeros((self.simplexDim+1,latent_points_arr.shape[1]))
        # dist_2_center = L2_dist(latent_points_arr,center)
        # dist_2_center[dist_2_center>radius] = 0
        # first_simplex_point = torch.max(dist_2_center,dim=0).indices
        # simplex_points[0,:] = latent_points_arr[first_simplex_point,:]
        # candidates = latent_points_arr[dist_2_center<radius]
        # for i_dim in range(1,simplexDim+1):
        #     L = 0
        #     for i_point in range(i_dim):
        #         L += L2_dist(candidates,simplex_points[i_point,:])
        #     farest_point = torch.max(L,dim=0).indices
        #     simplex_points[i_dim,:] = candidates[farest_point,:]

        #print("plot lattices")

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
        # interpolate_list = F.sigmoid(interpolate_list)
        fig, ax = plt.subplots(1,1)
        ax.set_xlim(0, int((n_side+1)*delta_coord))
        ax.set_ylim(0, int((n_side+1)*delta_coord))
        coordinatesList = [[0, 0], [100, 200], [200, 200]]
        cmap = 'Greens'
        cmap = plt.get_cmap(cmap) 
        for idx in range(n_total):
            tx, ty = coord_array[idx,0],coord_array[idx,1]
            ax.imshow(interpolate_list[idx].view(im_x, im_y).cpu().detach().numpy(),cmap=cmap, vmin=0, vmax=1.0,extent=(tx, tx + 28, ty, ty + 28))
            # ax[idx].imshow(interpolate_list[i_point].view(28, 28).cpu().detach().numpy(),cmap=cmap, vmin=0, vmax=1.0)
        ax.axis("off")    
        fig.savefig("./results/test_lattices_ternary",dpi = 450)





