#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Szu-Wei Fu

import torch
import torch.nn.functional as F
from .vector_quantize_pytorch import VectorQuantize

######## Model for speech enhancement
class CNN_1D_encoder_SE(torch.nn.Module):
    def __init__(self, codebook_dim):
        super().__init__()
        self.activation = torch.nn.LeakyReLU(negative_slope=0.3)
                
        self.conv_enc1 = torch.nn.Conv1d(in_channels=257, out_channels=200, kernel_size=7, stride=1, padding=3)
        self.conv_enc2 = torch.nn.Conv1d(in_channels=200, out_channels=200, kernel_size=7, stride=1, padding=3)
        self.conv_enc3 = torch.nn.Conv1d(in_channels=200, out_channels=150, kernel_size=7, stride=1, padding=3)
        self.conv_enc4 = torch.nn.Conv1d(in_channels=150, out_channels=150, kernel_size=7, stride=1, padding=3)
        self.conv_enc5 = torch.nn.Conv1d(in_channels=150, out_channels=codebook_dim, kernel_size=7, stride=1, padding=3)
        self.conv_enc6 = torch.nn.Conv1d(in_channels=codebook_dim, out_channels=codebook_dim, kernel_size=7, stride=1, padding=3)
        
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=codebook_dim, nhead=8, dim_feedforward=codebook_dim, dropout=0.4, 
                                                         activation='gelu', batch_first=True) # batch, seq, feature
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=2)
        
    def mean_removal(self, x):
        channel_mean = torch.mean(x, dim=-1, keepdim=True)
        return x-channel_mean
    
    def forward(self, x): # x.shape = torch.Size([B, T, 257])
        x = (x.transpose(2, 1)) # x.shape = torch.Size([B, 257, T])
        
        x = self.mean_removal(x) 
        enc1 = self.mean_removal(self.activation(self.conv_enc1(x))) 
        enc2 = self.mean_removal(self.activation(self.conv_enc2(enc1))) 
        enc3 = self.mean_removal(self.activation(self.conv_enc3(enc1+enc2))) 
        enc4 = self.mean_removal(self.activation(self.conv_enc4(enc3))) 
        enc5 = self.mean_removal(self.activation(self.conv_enc5(enc3+enc4))) 
        enc6 = self.mean_removal(self.activation(self.conv_enc6(enc5)))
        
        z = self.transformer_encoder((enc5+enc6).transpose(2, 1))
        z = self.mean_removal(z.transpose(2, 1))       
        return z

class CNN_1D_decoder_SE(torch.nn.Module):
    def __init__(self, codebook_dim):
        super().__init__()
        self.activation = torch.nn.LeakyReLU(negative_slope=0.3)
        
        self.conv_dec1 = torch.nn.Conv1d(in_channels=codebook_dim, out_channels=codebook_dim, kernel_size=7, stride=1, padding=3)
        self.conv_dec2 = torch.nn.Conv1d(in_channels=codebook_dim, out_channels=150, kernel_size=7, stride=1, padding=3)
        self.conv_dec3 = torch.nn.Conv1d(in_channels=150, out_channels=150, kernel_size=7, stride=1, padding=3)
        self.conv_dec4 = torch.nn.Conv1d(in_channels=150, out_channels=200, kernel_size=7, stride=1, padding=3)
        self.conv_dec5 = torch.nn.Conv1d(in_channels=200, out_channels=200, kernel_size=7, stride=1, padding=3)
        self.conv_dec6 = torch.nn.Conv1d(in_channels=200, out_channels=257, kernel_size=7, stride=1, padding=3)
        
        decoder_layer = torch.nn.TransformerEncoderLayer(d_model=codebook_dim, nhead=8, dim_feedforward=codebook_dim, dropout=0.4, 
                                                         activation='gelu', batch_first=True) # batch, seq, feature
        self.transformer_decoder = torch.nn.TransformerEncoder(decoder_layer, num_layers=2)
    
    def forward(self, zq): # x.shape = torch.Size([B, T, 128])
        zq = self.transformer_decoder(zq)
        
        dec1 = self.activation(self.conv_dec1(zq.transpose(2, 1)))
        dec2 = self.activation(self.conv_dec2(dec1))
        dec3 = self.activation(self.conv_dec3(dec2))
        dec4 = self.activation(self.conv_dec4(dec3+dec2))
        dec5 = self.activation(self.conv_dec5(dec4))
        out = F.relu(self.conv_dec6(dec5+dec4).transpose(2, 1))
        return out

class CNN_1D_quantizer_SE(torch.nn.Module):
    def __init__(self, codebook_size, codebook_dim, codebook_num, orthogonal_reg_weight, use_cosine_sim, ema_update, learnable_codebook,
                 stochastic_sample_codes, sample_codebook_temp, straight_through, reinmax, kmeans_init, threshold_ema_dead_code):
        super().__init__()
        
        self.quantizer = VectorQuantize(
            dim = codebook_dim,
            codebook_size = codebook_size,
            use_cosine_sim = use_cosine_sim,
            orthogonal_reg_weight = orthogonal_reg_weight,                 # in paper, they recommended a value of 10
            decay = 0.8,             # the exponential moving average decay, lower means the dictionary will change faster
            commitment_weight = 1,   # the weight on the commitment loss
            kmeans_init = kmeans_init,      # set to True
            kmeans_iters = 10,        # number of kmeans iterations to calculate the centroids for the codebook on init
            heads = codebook_num,   
            separate_codebook_per_head = True,
            ema_update = ema_update,
            learnable_codebook = learnable_codebook,
            stochastic_sample_codes = stochastic_sample_codes,
            sample_codebook_temp = sample_codebook_temp,
            straight_through = straight_through,
            reinmax = reinmax,
            threshold_ema_dead_code = threshold_ema_dead_code,
        )
        
    def forward(self, z, stochastic, update=True, indices=None): # x.shape = torch.Size([B, T, 257])
        if indices == None:
            zq, indices, vqloss, distance = self.quantizer(z.transpose(2, 1), stochastic, update=update)
            return zq, indices, vqloss, distance
        else:
            zq, cross_entropy_loss = self.quantizer(z.transpose(2, 1), stochastic, update=update, indices=indices)
            return zq, cross_entropy_loss


### VQVAE_SE ####
class VQVAE_SE(torch.nn.Module):
    def __init__(self, codebook_size, codebook_dim, codebook_num, orthogonal_reg_weight, use_cosine_sim, ema_update, learnable_codebook,
                 stochastic_sample_codes, sample_codebook_temp, straight_through, reinmax, kmeans_init, threshold_ema_dead_code):
        super().__init__()
        
        self.CNN_1D_encoder = CNN_1D_encoder_SE(codebook_dim)
        self.quantizer = CNN_1D_quantizer_SE(codebook_size, codebook_dim, codebook_num, orthogonal_reg_weight, use_cosine_sim, ema_update, learnable_codebook,
                     stochastic_sample_codes, sample_codebook_temp, straight_through, reinmax, kmeans_init, threshold_ema_dead_code)
        self.CNN_1D_decoder = CNN_1D_decoder_SE(codebook_dim)
        

######## Model for quality estimation
class CNN_1D_encoder_QE(torch.nn.Module):
    def __init__(self, codebook_dim):
        super().__init__()
        self.activation = torch.nn.LeakyReLU(negative_slope=0.3)
        
        # Normailization layer
        self.enc_In0 = torch.nn.InstanceNorm1d(257)
        self.enc_In1 = torch.nn.InstanceNorm1d(128)
        self.enc_In2 = torch.nn.InstanceNorm1d(128)
        self.enc_In3 = torch.nn.InstanceNorm1d(64)
        self.enc_In4 = torch.nn.InstanceNorm1d(64)
        self.enc_In5 = torch.nn.InstanceNorm1d(codebook_dim)
        self.enc_In6 = torch.nn.InstanceNorm1d(codebook_dim)
        self.enc_In7 = torch.nn.InstanceNorm1d(codebook_dim)
        
        ## Encoder
        self.conv_enc1 = torch.nn.Conv1d(in_channels=257, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.conv_enc2 = torch.nn.Conv1d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.conv_enc3 = torch.nn.Conv1d(in_channels=128, out_channels=64, kernel_size=7, stride=1, padding=3)
        self.conv_enc4 = torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=7, stride=1, padding=3)
        self.conv_enc5 = torch.nn.Conv1d(in_channels=64, out_channels=codebook_dim, kernel_size=7, stride=1, padding=3)
        self.conv_enc6 = torch.nn.Conv1d(in_channels=codebook_dim, out_channels=codebook_dim, kernel_size=7, stride=1, padding=3)
    
    def forward(self, x): # x.shape = torch.Size([B, T, 257])
        x = self.enc_In0(x.transpose(2, 1)) # x.shape = torch.Size([B, 257, T])
        
        enc1 = self.enc_In1(self.activation(self.conv_enc1(x))) # torch.Size([B, 128, T])
        enc2 = self.enc_In2(self.activation(self.conv_enc2(enc1))) # torch.Size([B, 128, T])
        enc3 = self.enc_In3(self.activation(self.conv_enc3(enc1+enc2))) # torch.Size([B, 64, T])
        enc4 = self.enc_In4(self.activation(self.conv_enc4(enc3))) # torch.Size([B, 64, T])
        enc5 = self.enc_In5(self.activation(self.conv_enc5(enc3+enc4))) # torch.Size([B, 32, T])
        z = self.enc_In6(self.conv_enc6(enc5)) # torch.Size([B, 32, T])            
        return z

class CNN_1D_decoder_QE(torch.nn.Module):
    def __init__(self, codebook_dim):
        super().__init__()
        self.activation = torch.nn.LeakyReLU(negative_slope=0.3)
         
        self.conv_dec1 = torch.nn.Conv1d(in_channels=codebook_dim, out_channels=codebook_dim, kernel_size=7, stride=1, padding=3)
        self.conv_dec2 = torch.nn.Conv1d(in_channels=codebook_dim, out_channels=64, kernel_size=7, stride=1, padding=3)
        self.conv_dec3 = torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=7, stride=1, padding=3)
        self.conv_dec4 = torch.nn.Conv1d(in_channels=64, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.conv_dec5 = torch.nn.Conv1d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.conv_dec6 = torch.nn.Conv1d(in_channels=128, out_channels=257, kernel_size=7, stride=1, padding=3)
        
    def forward(self, zq): # x.shape = torch.Size([B, T, 128])
        dec1 = (self.activation(self.conv_dec1(zq.transpose(2, 1))))
        dec2 = (self.activation(self.conv_dec2(dec1)))
        dec3 = (self.activation(self.conv_dec3(dec2)))
        dec4 = (self.activation(self.conv_dec4(dec3+dec2)))
        dec5 = (self.activation(self.conv_dec5(dec4)))
        out = F.relu(self.conv_dec6(dec5+dec4).transpose(2, 1)) # torch.Size([B, T, 257])
        return out

class CNN_1D_quantizer_QE(torch.nn.Module):
    def __init__(self, codebook_size, codebook_dim, codebook_num, orthogonal_reg_weight, use_cosine_sim, ema_update, learnable_codebook,
                 stochastic_sample_codes, sample_codebook_temp, straight_through, reinmax, kmeans_init, threshold_ema_dead_code, 
                 ):
        super().__init__()
        
        self.quantizer = VectorQuantize(
            dim = codebook_dim,
            codebook_size = codebook_size,
            use_cosine_sim = use_cosine_sim,
            orthogonal_reg_weight = orthogonal_reg_weight,                 # in paper, they recommended a value of 10
            decay = 0.8,             # the exponential moving average decay, lower means the dictionary will change faster
            commitment_weight = 1,   # the weight on the commitment loss
            kmeans_init = kmeans_init,      # set to True
            kmeans_iters = 10,        # number of kmeans iterations to calculate the centroids for the codebook on init
            heads = codebook_num,   
            separate_codebook_per_head = True,
            ema_update = ema_update,
            learnable_codebook = learnable_codebook,
            stochastic_sample_codes = stochastic_sample_codes,
            sample_codebook_temp = sample_codebook_temp,
            straight_through = straight_through,
            reinmax = reinmax,
            threshold_ema_dead_code = threshold_ema_dead_code
        )
        
    def forward(self, z, stochastic, update=True, indices = None): # x.shape = torch.Size([B, T, 257])
        if indices == None:
            zq, indices, vqloss, distance = self.quantizer(z.transpose(2, 1), stochastic, update)
            return zq, indices, vqloss, distance
        else:
            zq, cross_entropy_loss = self.quantizer(z.transpose(2, 1), stochastic, indices)
            return zq, cross_entropy_loss

### VQVAE_QE ####
class VQVAE_QE(torch.nn.Module):
    def __init__(self, codebook_size, codebook_dim, codebook_num, orthogonal_reg_weight, use_cosine_sim, ema_update, learnable_codebook,
                 stochastic_sample_codes, sample_codebook_temp, straight_through, reinmax, kmeans_init, threshold_ema_dead_code, 
                 ):
        super().__init__()
        
        self.CNN_1D_encoder = CNN_1D_encoder_QE(codebook_dim)
        self.quantizer = CNN_1D_quantizer_QE(codebook_size, codebook_dim, codebook_num, orthogonal_reg_weight, use_cosine_sim, ema_update, learnable_codebook,
                     stochastic_sample_codes, sample_codebook_temp, straight_through, reinmax, kmeans_init, threshold_ema_dead_code)
        self.CNN_1D_decoder = CNN_1D_decoder_QE(codebook_dim)
    


### use for inference
def resynthesize(enhanced_mag, noisy_inputs, hop_size):
    """Function for resynthesizing waveforms from enhanced mags.
    Arguments
    ---------
    enhanced_mag : torch.Tensor
        Predicted spectral magnitude, should be three dimensional.
    noisy_inputs : torch.Tensor
        The noisy waveforms before any processing, to extract phase.
    Returns
    -------
    enhanced_wav : torch.Tensor
        The resynthesized waveforms of the enhanced magnitudes with noisy phase.
    """

    # Extract noisy phase from inputs
        
    noisy_feats = torch.stft(noisy_inputs, n_fft=512, hop_length=hop_size, win_length=512, 
                             window=torch.hamming_window(512).to('cuda'),
                             center=True,
                             pad_mode="constant",
                             onesided=True,
                             return_complex=False).transpose(2, 1)
            
    noisy_phase = torch.atan2(noisy_feats[:, :, :, 1], noisy_feats[:, :, :, 0])[:,0:enhanced_mag.shape[1],:]
    
    # Combine with enhanced magnitude
    predictions = torch.mul(
        torch.unsqueeze(enhanced_mag, -1),
        torch.cat(
            (
                torch.unsqueeze(torch.cos(noisy_phase), -1),
                torch.unsqueeze(torch.sin(noisy_phase), -1),
            ),
            -1,
        ),
    ).permute(0, 2, 1, 3)
    
    # isft ask complex input
    complex_predictions = torch.complex(predictions[..., 0], predictions[..., 1])
    pred_wavs = torch.istft(input=complex_predictions, n_fft=512, hop_length=hop_size, win_length=512, 
                            window=torch.hamming_window(512).to('cuda'),
                            center=True,
                            onesided=True,
                            length=noisy_inputs.shape[1])
    
    return pred_wavs

def stft_magnitude(x, hop_size, fft_size=512, win_length=512):
    if x.is_cuda:
        x_stft = torch.stft(
            x, fft_size, hop_size, win_length, window=torch.hann_window(win_length).to('cuda'), return_complex=False
        )
    else:
        x_stft = torch.stft(
            x, fft_size, hop_size, win_length, window=torch.hann_window(win_length), return_complex=False
        )   
    real = x_stft[..., 0]
    imag = x_stft[..., 1]
    
    return torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7)).transpose(2, 1)

def cos_loss(SP_noisy, SP_y_noisy):  
    eps=1e-5
    SP_noisy_norm = torch.norm(SP_noisy, p=2, dim=-1, keepdim=True)+eps
    SP_y_noisy_norm = torch.norm(SP_y_noisy, p=2, dim=-1, keepdim=True)+eps  
    Cos_frame = torch.sum(SP_noisy/SP_noisy_norm * SP_y_noisy/SP_y_noisy_norm, dim=-1) # torch.Size([B, T, 1])
    
    return -torch.mean(Cos_frame)