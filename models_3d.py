import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
from functools import reduce
import operator

class Model(nn.Module):
    def __init__(self,output_dim, hidden_dim, latent_dim,device,model_type,im_x,im_y,im_z,modes1,modes2,modes3):
        super(Model, self).__init__()
        self.device = device
        self.im_x = im_x
        self.im_y = im_y
        self.im_z = im_z
        self.model_type = model_type
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        if model_type == 'FNO3D':
            self.Encoder = FNO3D_Encoder(input_dim=4, hidden_dim=hidden_dim, latent_dim=latent_dim,im_x=im_x, im_y=im_y, im_z=im_z, modes1=modes1, modes2=modes2, modes3=modes3)
            self.Decoder = FNO3D_Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=output_dim, im_x=im_x, im_y=im_y, im_z=im_z, modes1=modes1, modes2=modes2, modes3=modes3)
        elif model_type == 'Freq_FNO3D':
            self.Encoder = FNO3D_Encoder(input_dim=4, hidden_dim=hidden_dim, latent_dim=latent_dim,im_x=im_x, im_y=im_y, im_z=im_z, modes1=modes1, modes2=modes2, modes3=modes3)
            self.Decoder = FreqFNO3D_Decoder(latent_dim=latent_dim//2, hidden_dim=hidden_dim,output_dim=output_dim,im_x=im_x, im_y=im_y, im_z=im_z, modes1=modes1, modes2=modes2, modes3=modes3)

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)
        z = mean + var*epsilon                    
        return z
    
    def reparameterization_FreqNO3D(self,mean,var):
        # latent_dim = mean.shape[1]
        # epsilon = torch.randn(mean.shape[0], latent_dim, self.modes1, self.modes2,dtype = torch.complex64).to(self.device)
        # z = mean + epsilon* torch.sqrt(var)
        modes1 = self.modes1
        modes2 = self.modes2
        modes3 = self.modes3
        latent_dim = mean.shape[1] 
        epsilon_real = torch.randn(mean.shape[0], latent_dim//2, modes1, modes2, modes3).to(self.device)* torch.sqrt(var[:,:latent_dim//2,:,:])
        z_real = mean[:,:latent_dim//2,:,:] + epsilon_real
        epsilon_image = torch.randn(mean.shape[0], latent_dim//2, modes1, modes2, modes3).to(self.device)* torch.sqrt(var[:,latent_dim//2:,:,:])
        z_image = mean[:,latent_dim//2:,:,:] + epsilon_image
        z = torch.complex(z_real, z_image)
        return z
    
    def forward(self, x):
        if self.model_type == 'FNO3D':
            mean, var = self.Encoder(x)
            z = self.reparameterization(mean, torch.sqrt(var))
            x_hat = self.Decoder(z)
            return x_hat, mean, var
        elif self.model_type == "Freq_FNO3D":
            mean, var = self.Encoder(x)
            z = self.reparameterization_FreqNO3D(mean, var)
            x_hat = self.Decoder(z,output_im_x=self.im_x, output_im_y=self.im_y, output_im_z=self.im_z)
            return x_hat, mean, var
        else:
            mean, log_var = self.Encoder(x)
            z = self.reparameterization(mean, torch.exp(0.5 * log_var)) 
            x_hat = self.Decoder(z)
            return x_hat, mean, log_var    

class FNO3D_Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, im_x, im_y, im_z, modes1, modes2, modes3):
        super(FNO3D_Encoder, self).__init__()
        self.im_x = im_x
        self.im_y = im_y
        self.im_z = im_z
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.p = nn.Linear(input_dim, hidden_dim)  # +3 for (x, y, z)

        self.conv0 = SpectralConv3d(hidden_dim, hidden_dim, modes1, modes2, modes3)
        self.conv1 = SpectralConv3d(hidden_dim, hidden_dim, modes1, modes2, modes3)
        self.conv2 = SpectralConv3d(hidden_dim, latent_dim, modes1, modes2, modes3)

        self.mlp0 = MLP3D(hidden_dim, hidden_dim, hidden_dim)
        self.mlp1 = MLP3D(hidden_dim, hidden_dim, hidden_dim)
        self.mlp2 = MLP3D(latent_dim, latent_dim, latent_dim)

        self.w0 = nn.Conv3d(hidden_dim, hidden_dim, 1)
        self.w1 = nn.Conv3d(hidden_dim, hidden_dim, 1)
        self.w2 = nn.Conv3d(hidden_dim, latent_dim, 1)

        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):  # x: (B, D, H, W, C)
        grid = self.get_grid(x.shape, x.device)
        #print("x shape: {}, grid shape: {}".format(x.shape, grid.shape))
        x = torch.cat((x, grid), dim=-1)  # (B, D, H, W, C+3)
        x = self.p(x)  # (B, D, H, W, hidden_dim)
        x = x.permute(0, 4, 1, 2, 3)  # (B, hidden_dim, D, H, W)

        x1 = self.conv0(x)
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = self.activation(x1 + x2)

        x1 = self.conv1(x)
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = self.activation(x1 + x2)

        x1 = self.conv2(x)
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2

        mean = torch.mean(x, dim=(2, 3, 4), keepdim=True)
        var = torch.var(x, dim=(2, 3, 4), keepdim=True)
        return mean, var

    def get_grid(self, shape, device):
        B, D, H, W, _ = shape
        gridx = torch.linspace(0, 1, self.im_x, dtype=torch.float32, device=device).view(1, self.im_x, 1, 1, 1).repeat(B, 1, self.im_y, self.im_z, 1)
        gridy = torch.linspace(0, 1, self.im_y, dtype=torch.float32, device=device).view(1, 1, self.im_y, 1, 1).repeat(B, self.im_x, 1, self.im_z, 1)
        gridz = torch.linspace(0, 1, self.im_z, dtype=torch.float32, device=device).view(1, 1, 1, self.im_z, 1).repeat(B, self.im_x, self.im_y, 1, 1)
        return torch.cat((gridx, gridy, gridz), dim=-1)


class FNO3D_Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, im_x, im_y, im_z, modes1, modes2, modes3):
        super(FNO3D_Decoder, self).__init__()
        self.im_x = im_x
        self.im_y = im_y
        self.im_z = im_z
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.p = LocalMLP3D(latent_dim, hidden_dim, im_x, im_y, im_z)

        self.conv0 = SpectralConv3d(hidden_dim, hidden_dim, modes1, modes2, modes3)
        self.conv1 = SpectralConv3d(hidden_dim, hidden_dim, modes1, modes2, modes3)
        self.conv2 = SpectralConv3d(hidden_dim, hidden_dim, modes1, modes2, modes3)
        self.conv3 = SpectralConv3d(hidden_dim, hidden_dim, modes1, modes2, modes3)

        self.mlp0 = MLP3D(hidden_dim, hidden_dim, hidden_dim * 2)
        self.mlp1 = MLP3D(hidden_dim, hidden_dim, hidden_dim * 2)
        self.mlp2 = MLP3D(hidden_dim, hidden_dim, hidden_dim * 2)
        self.mlp3 = MLP3D(hidden_dim, hidden_dim, hidden_dim * 2)

        self.w0 = nn.Conv3d(hidden_dim, hidden_dim, 1)
        self.w1 = nn.Conv3d(hidden_dim, hidden_dim, 1)
        self.w2 = nn.Conv3d(hidden_dim, hidden_dim, 1)
        self.w3 = nn.Conv3d(hidden_dim, hidden_dim, 1)

        self.q = MLP3D(hidden_dim, 1, latent_dim)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.activation(self.p(x))  # (B, hidden_dim, D, H, W)

        x1 = self.conv0(x)
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = self.activation(x1 + x2)

        x1 = self.conv1(x)
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = self.activation(x1 + x2)

        x1 = self.conv2(x)
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = self.activation(x1 + x2)

        x1 = self.conv3(x)
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2

        x = self.q(x)
        x = torch.sigmoid(x)
        x = x.permute(0, 2, 3, 4, 1)  # (B, D, H, W, output_channels)
        return x
    
class FreqFNO3D_Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim,im_x,im_y,im_z, modes1,modes2, modes3):
        super(FreqFNO3D_Decoder, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.im_x = im_x
        self.im_y = im_y
        self.im_z = im_z
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.p = LocalMLP3D_Complex(in_channels=latent_dim, out_channels=hidden_dim,modes1=modes1, modes2=modes2, modes3=modes3)

        self.conv0 = SpectralConv3d(self.hidden_dim, self.hidden_dim, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.hidden_dim, self.hidden_dim, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.hidden_dim, self.hidden_dim, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.hidden_dim, self.hidden_dim, self.modes1, self.modes2, self.modes3)
        self.mlp0 = MLP3D(self.hidden_dim, self.hidden_dim, self.hidden_dim*2)
        self.mlp1 = MLP3D(self.hidden_dim, self.hidden_dim, self.hidden_dim*2)
        self.mlp2 = MLP3D(self.hidden_dim, self.hidden_dim, self.hidden_dim*2)
        self.mlp3 = MLP3D(self.hidden_dim, self.hidden_dim, self.hidden_dim*2)
        self.w0 = nn.Conv3d(self.hidden_dim, self.hidden_dim, 1)
        self.w1 = nn.Conv3d(self.hidden_dim, self.hidden_dim, 1)
        self.w2 = nn.Conv3d(self.hidden_dim, self.hidden_dim, 1)
        self.w3 = nn.Conv3d(self.hidden_dim, self.hidden_dim, 1)
        self.q = MLP3D(hidden_dim, 1, latent_dim) # output channel is 1: u(x, y, z)
        self.complex_activation_function = ComplexReLU(0.2)
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, x, output_im_x = 64, output_im_y = 64, output_im_z = 64):
        x = self.complex_activation_function(self.p(x))

        x = torch.fft.irfft2(x, s=(output_im_x, output_im_y, output_im_z),dim=(-3,-2,-1))

        x1 = self.conv0(x)
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = self.LeakyReLU(x)

        x1 = self.conv1(x)
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = self.LeakyReLU(x)

        x1 = self.conv2(x)
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = self.LeakyReLU(x)

        x1 = self.conv3(x)
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2
        x = self.q(x)
        x = torch.sigmoid(x)
        x = x.permute(0, 2, 3, 4, 1)
        return x

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1), dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x

class MLP3D(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP3D, self).__init__()
        self.mlp1 = nn.Conv3d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv3d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x

class LocalMLP3D(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(LocalMLP3D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.float32)
        )

    def compl_mul3d(self, input, weights):
        # input: (B, in_channels, D, H, W)
        # weights: (in_channels, out_channels, D, H, W)
        # output: (B, out_channels, D, H, W)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        return self.compl_mul3d(x, self.weights)

    
class LocalMLP3D_Complex(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(LocalMLP3D_Complex, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
    
    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        out = self.compl_mul3d(x, self.weights1)
        return out

class ComplexReLU(nn.Module):
    def __init__(self, negative_slope):
        super(ComplexReLU, self).__init__()
        self.negative_slope = negative_slope
    def forward(self, x):
        LeakyReLU = nn.LeakyReLU(self.negative_slope)
        return torch.complex(LeakyReLU(x.real), LeakyReLU(x.imag))
    
class ComplexTanh(nn.Module):
    def __init__(self):
        super(ComplexTanh, self).__init__()
    def forward(self, x):
        return torch.complex(F.tanh(x.real), F.tanh(x.imag))