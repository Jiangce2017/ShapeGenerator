import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    def __init__(self,x_dim, hidden_dim, latent_dim,device,model_type,im_x,im_y,modes1, modes2):
        super(Model, self).__init__()
        self.device = device
        self.im_x = im_x
        self.im_y = im_y
        self.model_type = model_type
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.modes1 = modes1
        self.modes2 = modes2
        if model_type == 'CNN':
            self.Encoder = CNN_Encoder(input_dim=1, hidden_dim=hidden_dim, latent_dim=latent_dim,im_x=im_x, im_y=im_y)
            self.Decoder = CNN_Decoder(latent_dim=latent_dim, hidden_dim = hidden_dim, output_dim = x_dim,im_x=im_x, im_y=im_y)
        elif model_type == 'FL':
            self.Encoder = FL_Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
            self.Decoder = FL_Decoder(latent_dim=latent_dim, hidden_dim = hidden_dim, output_dim = x_dim)
        elif model_type == 'FNO':
            self.Encoder = FNO_Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim,im_x=im_x, im_y=im_y,modes1=modes1,modes2=modes2)
            self.Decoder = FNO_Decoder(latent_dim=latent_dim, hidden_dim = hidden_dim, output_dim = x_dim,im_x=im_x, im_y=im_y,modes1=modes1,modes2=modes2)
        elif model_type == 'Freq_FNO':
            self.Encoder = FNO_Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim,im_x=im_x, im_y=im_y,modes1=modes1,modes2=modes2)
            self.Decoder = FreqFNO_Decoder(latent_dim=latent_dim//2, hidden_dim = hidden_dim, output_dim = x_dim,im_x=im_x, im_y=im_y,modes1=10,modes2=6)
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)
        z = mean + var*epsilon                    
        return z

    def reparameterization_NO(self,mean,var):
        epsilon = torch.randn(mean.shape[0], self.latent_dim, self.im_x, self.im_y).to(self.device)
        z = mean + epsilon* torch.sqrt(var)
        return z
    
    def reparameterization_FreqNO(self,mean,var):
        # latent_dim = mean.shape[1]
        # epsilon = torch.randn(mean.shape[0], latent_dim, self.modes1, self.modes2,dtype = torch.complex64).to(self.device)
        # z = mean + epsilon* torch.sqrt(var)
        modes1 = 10
        modes2 = 6
        latent_dim = mean.shape[1] 
        epsilon_real = torch.randn(mean.shape[0], latent_dim//2, modes1, modes2).to(self.device)* torch.sqrt(var[:,:latent_dim//2,:,:])
        z_real = mean[:,:latent_dim//2,:,:] + epsilon_real
        epsilon_image = torch.randn(mean.shape[0], latent_dim//2, modes1, modes2).to(self.device)* torch.sqrt(var[:,latent_dim//2:,:,:])
        z_image = mean[:,latent_dim//2:,:,:] + epsilon_image
        z = torch.complex(z_real, z_image)
        return z
    
    def forward(self, x):
        if self.model_type == 'FNO':
            mean, var= self.Encoder(x)
            z = self.reparameterization_NO(mean, var)
            x_hat = self.Decoder(z)
            return x_hat, mean, var
        elif self.model_type == 'Freq_FNO':
            mean, var= self.Encoder(x)
            z = self.reparameterization_FreqNO(mean, var)
            x_hat = self.Decoder(z)
            return x_hat, mean, var
        else:
            mean, log_var = self.Encoder(x)
            z = self.reparameterization(mean, torch.exp(0.5 * log_var)) 
            x_hat = self.Decoder(z)
            return x_hat, mean, log_var    

class FL_Encoder(nn.Module):
        def __init__(self, input_dim, hidden_dim, latent_dim):
            super(FL_Encoder, self).__init__()
            self.FC_input = nn.Linear(input_dim, hidden_dim)
            self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
            self.FC_mean  = nn.Linear(hidden_dim, latent_dim)
            self.FC_var   = nn.Linear (hidden_dim, latent_dim)
            self.LeakyReLU = nn.LeakyReLU(0.2)
            self.training = True
            self.input_dim = input_dim
            
        def forward(self, x):
            x = x.view(-1, self.input_dim)
            h_       = self.LeakyReLU(self.FC_input(x))
            h_       = self.LeakyReLU(self.FC_input2(h_))
            mean     = self.FC_mean(h_)
            log_var  = self.FC_var(h_)                                                                
            return mean, log_var
        
class FL_Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(FL_Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        h     = self.LeakyReLU(self.FC_hidden(x))
        h     = self.LeakyReLU(self.FC_hidden2(h))
        x_hat = torch.sigmoid(self.FC_output(h))
        return x_hat
    
class CNN_Encoder(nn.Module):
        def __init__(self, input_dim, hidden_dim, latent_dim,im_x,im_y):
            super(CNN_Encoder, self).__init__()
            self.conv1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=(3, 3), stride=1, padding=1)
            self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3, 3), stride=1, padding=1)
            self.maxpool = nn.MaxPool2d(kernel_size=(2, 2)) ## half spatial dimension 
            self.conv3 = nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=(3, 3), stride=1, padding=1)
            self.conv4 = nn.Conv2d(hidden_dim*2, hidden_dim*2, kernel_size=(3, 3), stride=1, padding=1)
            self.conv5 = nn.Conv2d(hidden_dim*2, hidden_dim*2, kernel_size=(3, 3), stride=1, padding=1)
            self.flatten = nn.Flatten()
            self.dense1 = nn.Linear(im_x*im_y*hidden_dim//2, hidden_dim)
            self.layer_mean = nn.Linear(hidden_dim, latent_dim)
            self.layer_variance = nn.Linear(hidden_dim, latent_dim)
            self.LeakyReLU = nn.LeakyReLU(0.2)
            self.im_x = im_x
            self.im_y = im_y

        def forward(self, x):
            x = x.view(-1,1,self.im_x,self.im_y)
            h_ = self.LeakyReLU(self.conv1(x))
            h_ = self.LeakyReLU(self.conv2(h_))
            h_ = self.maxpool(h_)
            h_ = self.LeakyReLU(self.conv3(h_))
            h_ = self.LeakyReLU(self.conv4(h_))
            h_ = self.LeakyReLU(self.conv5(h_))
            h_ = self.flatten(h_)
            h_ = self.LeakyReLU(self.dense1(h_))
            mean     = self.layer_mean(h_)
            log_var  = self.layer_variance(h_)                                                                           
            return mean, log_var

class CNN_Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim,im_x,im_y):
        super(CNN_Decoder, self).__init__()

        self.dense1 = nn.Linear(latent_dim, im_x*im_y*2)
        self.dense2 = nn.Linear(im_x*im_y*2,im_x*im_y*hidden_dim//2)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv1 = nn.Conv2d(hidden_dim*2, hidden_dim, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 1, kernel_size=(3, 3), stride=1, padding=1)

        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()
        self.hidden_dim = hidden_dim
        self.im_x = im_x
        self.im_y = im_y

    def forward(self, x):
        h = self.LeakyReLU(self.dense1(x))
        h = self.LeakyReLU(self.dense2(h))
        h = h.view(-1,self.hidden_dim*2,self.im_x//2,self.im_y//2)
        h = self.upsample(h)
        h = self.LeakyReLU(self.conv1(h))
        x_hat = self.sigmoid(self.conv2(h))
        return x_hat    

class FNO_Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim,im_x,im_y,modes1, modes2):
        super(FNO_Encoder, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.im_x = im_x
        self.im_y = im_y
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.activation_function = nn.LeakyReLU(0.2)
        #self.activation_function = F.tanh
        self.p = nn.Linear(3, self.hidden_dim) # input channel is 3: (a(x, y), x, y)
        self.conv0 = SpectralConv2d(self.hidden_dim, self.hidden_dim, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.hidden_dim, self.hidden_dim, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.hidden_dim, self.latent_dim, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.latent_dim, self.latent_dim, self.modes1, self.modes2)
        self.conv4 = SpectralConv2d(self.latent_dim, self.latent_dim, self.modes1, self.modes2)
        self.conv5 = SpectralConv2d(self.latent_dim, self.latent_dim, self.modes1, self.modes2)
        self.mlp0 = MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim)
        self.mlp1 = MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim)
        self.mlp2 = MLP(self.latent_dim, self.latent_dim, self.latent_dim)
        self.mlp3 = MLP(self.latent_dim, self.latent_dim, self.latent_dim)
        self.mlp4 = MLP(self.latent_dim, self.latent_dim, self.latent_dim)
        self.mlp5 = MLP(self.latent_dim, self.latent_dim, self.latent_dim)
        self.w0 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 1)
        self.w1 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 1)
        self.w2 = nn.Conv2d(self.hidden_dim, self.latent_dim, 1)
        self.w3 = nn.Conv2d(self.latent_dim, self.latent_dim, 1)
        self.w4 = nn.Conv2d(self.latent_dim, self.latent_dim, 1)
        self.w5 = nn.Conv2d(self.latent_dim, self.latent_dim, 1)

    def forward(self, x):
        x = x.view(-1,self.im_x,self.im_y,1)
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.activation_function(self.p(x))
        x = x.permute(0, 3, 1, 2)

        x1 = self.conv0(x)
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = self.activation_function(x)

        x1 = self.conv1(x)
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = self.activation_function(x)

        x1 = self.conv2(x)
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = self.activation_function(x)

        x1 = self.conv3(x)
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2
        x = self.activation_function(x)

        x1 = self.conv4(x)
        x1 = self.mlp4(x1)
        x2 = self.w4(x)
        x = x1 + x2
        x = self.activation_function(x)

        x1 = self.conv5(x)
        x1 = self.mlp5(x1)
        x2 = self.w5(x)
        x = x1 + x2

        mean = torch.mean(x,dim=(2,3),keepdim=True)
        var = torch.var(x,dim=(2,3),keepdim=True)
        return mean, var
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)
    
class FNO_Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim,im_x,im_y,modes1,modes2):
        super(FNO_Decoder, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.im_x = im_x
        self.im_y = im_y
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.p = LocalMLP(latent_dim,hidden_dim, self.im_x, self.im_y)
        self.conv0 = SpectralConv2d(self.hidden_dim, self.hidden_dim, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.hidden_dim, self.hidden_dim, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.hidden_dim, self.hidden_dim, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.hidden_dim, self.hidden_dim, self.modes1, self.modes2)
        self.mlp0 = MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim*2)
        self.mlp1 = MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim*2)
        self.mlp2 = MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim*2)
        self.mlp3 = MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim*2)
        self.w0 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 1)
        self.w1 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 1)
        self.w2 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 1)
        self.w3 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 1)
        self.q = MLP(self.hidden_dim, 1, self.latent_dim) # output channel is 1: u(x, y)
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        x = self.LeakyReLU(self.p(x))
        
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
        x = 1/(1+torch.exp(-32*x))
        x = x.permute(0, 2, 3, 1)

        return x
    
class FreqFNO_Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim,im_x,im_y,modes1,modes2):
        super(FreqFNO_Decoder, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.im_x = im_x
        self.im_y = im_y
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.p = LocalMLP_Complex(latent_dim,hidden_dim, self.modes1, self.modes2)
        self.conv0 = SpectralConv2d(self.hidden_dim, self.hidden_dim, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.hidden_dim, self.hidden_dim, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.hidden_dim, self.hidden_dim, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.hidden_dim, self.hidden_dim, self.modes1, self.modes2)
        self.mlp0 = MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim*2)
        self.mlp1 = MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim*2)
        self.mlp2 = MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim*2)
        self.mlp3 = MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim*2)
        self.w0 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 1)
        self.w1 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 1)
        self.w2 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 1)
        self.w3 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 1)
        self.q = MLP(self.hidden_dim, 1, self.latent_dim) # output channel is 1: u(x, y)
        self.complex_activation_function = ComplexReLU(0.2)
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, x, output_im_x = 50, output_im_y = 50):
        x = self.complex_activation_function(self.p(x))

        x = torch.fft.irfft2(x, s=(output_im_x, output_im_y),dim=(-2,-1))

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
        x = F.sigmoid(x)
        x = x.permute(0, 2, 3, 1)
        return x

################################################################
# fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x

class LocalMLP(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(LocalMLP, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.float32))
    
    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        out = self.compl_mul2d(x, self.weights1)
        return out
    
class LocalMLP_Complex(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(LocalMLP_Complex, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.complex64))
    
    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        out = self.compl_mul2d(x, self.weights1)
        return out
    
class MLP_Complex(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP_Complex, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1, dtype=torch.complex64)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1, dtype=torch.complex64)
        self.LeakyReLU = ComplexReLU(0.2)

    def forward(self, x):
        x = self.mlp1(x)
        x = self.LeakyReLU(x)
        x = self.mlp2(x)
        return x
    
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
    