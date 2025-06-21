import torch
import torch.nn as nn
import torch.nn.functional as F

# # Model(x_dim, hidden_dim, latent_dim, device, model_type, im_x, im_y)

class Model(nn.Module):
    def __init__(self,x_dim, hidden_dim, latent_dim,device,model_type,im_x,im_y):
        super(Model, self).__init__()
        self.device = device
        if model_type == 'CNN':
            self.Encoder = CNN_Encoder(input_dim=1, hidden_dim=hidden_dim, latent_dim=latent_dim,im_x=im_x, im_y=im_y)
            self.Decoder = CNN_Decoder(latent_dim=latent_dim, hidden_dim = hidden_dim, output_dim = x_dim,im_x=im_x, im_y=im_y)
        elif model_type == 'FL':
            self.Encoder = FL_Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
            self.Decoder = FL_Decoder(latent_dim=latent_dim, hidden_dim = hidden_dim, output_dim = x_dim)
        elif model_type == 'FNO':
            self.Encoder = FNO_Encoder(input_dim=1, hidden_dim=hidden_dim, latent_dim=latent_dim, im_x=im_x, im_y=im_y)
            self.Decoder = FNO_Decoder(latent_dim=latent_dim, hidden_dim = hidden_dim, output_dim = 1, im_x=im_x, im_y=im_y)
        
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)        # sampling epsilon        
        z = mean + var*epsilon                          # reparameterization trick
        return z
        
    def forward(self, x):
        mean, log_var = self.Encoder(x)
        log_var = torch.clamp(log_var, min=-4, max=4)  # add this line

        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)
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
            log_var  = self.FC_var(h_)                     # encoder produces mean and log of variance 
                                                        #             (i.e., parateters of simple tractable normal distribution "q"
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
            self.dense1 = nn.Linear(im_x*im_y*16, hidden_dim)
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
            log_var  = self.layer_variance(h_)                     # encoder produces mean and log of variance 
                                                        #             (i.e., parateters of simple tractable normal distribution "q"
            return mean, log_var
    
class CNN_Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim,im_x,im_y):
        super(CNN_Decoder, self).__init__()

        self.dense1 = nn.Linear(latent_dim, im_x*im_y*2)
        self.dense2 = nn.Linear(im_x*im_y*2,im_x*im_y*16)

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
        def __init__(self, input_dim, hidden_dim, latent_dim, im_x, im_y, freq_filter_x = 50, freq_filter_y = 26):
            super(FNO_Encoder, self).__init__()
            self.hidden_dim = hidden_dim
            self.im_x = im_x
            self.im_y = im_y
            self.freq_filter_x = freq_filter_x
            self.freq_filter_y = freq_filter_y

            # Up projection (1×1)
            self.input_proj = nn.Conv2d(input_dim, hidden_dim, kernel_size=1)

            # Learnable real and imaginary frequency weights
            self.weight_real = nn.Parameter(torch.randn(hidden_dim, hidden_dim, freq_filter_x, freq_filter_y))
            self.weight_imag = nn.Parameter(torch.randn(hidden_dim, hidden_dim, freq_filter_x, freq_filter_y))

            # Second set of frequency weights for second FNO block
            self.weight_real_2 = nn.Parameter(torch.randn(hidden_dim, hidden_dim, freq_filter_x, freq_filter_y))
            self.weight_imag_2 = nn.Parameter(torch.randn(hidden_dim, hidden_dim, freq_filter_x, freq_filter_y))


            # Flatten and latent parameter layers
            self.flatten = nn.Flatten()
            self.layer_mean = nn.Linear(hidden_dim * im_x * im_y, latent_dim)
            self.layer_variance = nn.Linear(hidden_dim * im_x * im_y, latent_dim)

            
        def forward(self, x):
            x = x.view(-1,1,self.im_x,self.im_y)

            x = self.input_proj(x)
            x_ft = torch.fft.rfft2(x, norm='ortho') 

            real = x_ft.real[:, :, :self.freq_filter_x, :self.freq_filter_y]
            imag = x_ft.imag[:, :, :self.freq_filter_x, :self.freq_filter_y]

            # Matrix multiply each (real, imag) slice with learnable weights
            real_out = torch.einsum("bchw,cdhw->bdhw", real, self.weight_real) \
                    - torch.einsum("bchw,cdhw->bdhw", imag, self.weight_imag)
            imag_out = torch.einsum("bchw,cdhw->bdhw", real, self.weight_imag) \
                    + torch.einsum("bchw,cdhw->bdhw", imag, self.weight_real)

            # Reconstruct full frequency domain with filtered values
            x_ft = torch.complex(real_out, imag_out)

            x = torch.fft.irfft2(x_ft, s=(self.im_x, self.im_y), norm='ortho')
            x = F.relu(x) 

            x_flat = self.flatten(x)  # shape: [B, hidden_dim * im_x * im_y]
            mean = self.layer_mean(x_flat)
            log_var = self.layer_variance(x_flat)

            return mean, log_var
    
class FNO_Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, im_x, im_y, freq_filter_x = 50, freq_filter_y = 26):
        super(FNO_Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.im_x = im_x
        self.im_y = im_y
        self.freq_filter_x = freq_filter_x

        # Map latent vector to full feature map
        self.latent_to_feature = nn.Linear(latent_dim, hidden_dim * im_x * im_y)

        # Learnable frequency weights (real and imaginary)
        self.weight_real = nn.Parameter(torch.randn(hidden_dim, hidden_dim, freq_filter_x, freq_filter_y))
        self.weight_imag = nn.Parameter(torch.randn(hidden_dim, hidden_dim, freq_filter_x, freq_filter_y))

        self.weight_real_2 = nn.Parameter(torch.randn(hidden_dim, hidden_dim, freq_filter_x, freq_filter_y))
        self.weight_imag_2 = nn.Parameter(torch.randn(hidden_dim, hidden_dim, freq_filter_x, freq_filter_y))


        # Down projection: map back to output_dim (e.g., 1 for grayscale image)
        self.output_proj = nn.Conv2d(hidden_dim, output_dim, kernel_size=1)
        
    def forward(self, x):
        # Step 1: Linear projection from latent vector to full feature map
        x = self.latent_to_feature(x)  # shape: [B, hidden_dim * im_x * im_y]
        x = x.view(-1, self.hidden_dim, self.im_x, self.im_y)  # shape: [B, hidden_dim, H, W]

        # Step 2: Apply frequency domain weights
        x_ft = torch.fft.rfft2(x, norm='ortho')

        real = x_ft.real[:, :, :self.freq_filter_x, :self.freq_filter_y]
        imag = x_ft.imag[:, :, :self.freq_filter_x, :self.freq_filter_y]

        real_out = torch.einsum("bchw,cdhw->bdhw", real, self.weight_real) \
                 - torch.einsum("bchw,cdhw->bdhw", imag, self.weight_imag)
        imag_out = torch.einsum("bchw,cdhw->bdhw", real, self.weight_imag) \
                 + torch.einsum("bchw,cdhw->bdhw", imag, self.weight_real)

        x_ft = torch.complex(real_out, imag_out)

        # Step 3: Inverse FFT to return to spatial domain
        x = torch.fft.irfft2(x_ft, s=(self.im_x, self.im_y), norm='ortho')
        x = F.relu(x) 

        # Step 4: Project down to output channel (e.g., grayscale image)
        x_hat = self.output_proj(x)
        x_hat = torch.sigmoid(x_hat)

        return x_hat    
