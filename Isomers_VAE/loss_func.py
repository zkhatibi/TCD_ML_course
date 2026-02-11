import torch.nn as nn
import torch

ce_loss_fn = nn.CrossEntropyLoss(reduction='mean')  # Sum over batch and seq
mse_loss_fn = nn.MSELoss(reduction='mean')

def vae_loss(recon_x, x, prop, prediction, logvar, mu, KLD_weight):
    recon_loss = ce_loss_fn(recon_x, x)
    
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) # sums over the batch of data, excludes sum over the seq len
    kl_loss/= x.size(0)
    regress_loss = mse_loss_fn(prediction, prop.view(-1, 1))
    return recon_loss + KLD_weight * kl_loss