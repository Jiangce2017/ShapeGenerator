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
from utils import loss_function, load_mat, show_image, CombinedDataset,Logger,plot_ternary

if __name__ == '__main__':
    cuda = False
    device = torch.device("cuda" if cuda else "cpu")
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
        
    model_file = osp.join("checkpoints","gpu_"+model_type+"_model.pth")
    loaded_model = torch.load(model_file,map_location=torch.device('cpu'))
    loaded_model.device = device
    loaded_model = loaded_model.to(device)
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
    ternary_output_file =  "./results/test_lattices_ternary"
    plot_ternary(simplex_points,loaded_model,im_x, im_y,latent_dim, ternary_output_file)
    




