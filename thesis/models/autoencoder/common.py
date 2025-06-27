import math

import numpy as np
import torch 
import torch.nn.functional as F 


class AutoencoderLoss(torch.nn.Module):
    def __init__(self, kl_weight=1e-6):
        super().__init__()
        self.kl_weight = kl_weight
    
    def recon_kl_loss(
        self, inputs, reconstructions, posteriors, split="train"
    ):
        rec_loss = torch.nn.functional.mse_loss(inputs, reconstructions)
        
        # kl_loss = posteriors.kl()
        # kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        
        kl_loss = posteriors.kl(other='FreeBits', lambda_=0.5)
        kl_loss = kl_loss.mean()

        loss = rec_loss + self.kl_weight * kl_loss

        log = {
            "{}/ae_total_loss".format(split): loss.clone().detach().mean(),
            "{}/kl_loss".format(split): kl_loss.detach().mean(),
            "{}/rec_loss".format(split): rec_loss.detach().mean(),
        }
        return loss, log
    
class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=-1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)
        self.beta = 0.5

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None, lambda_=0):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other == None:
                return self.beta * torch.sum(torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=[1,2])
            elif other == 'FreeBits': 
                kl_dim = torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar  # (B, 1, latent_dim)
                kl = torch.clamp_min(kl_dim, lambda_)
                return self.beta * kl.sum(dim=[1, 2]) # (B)    
            else:
                return self.beta * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean

def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight
    
def frange_cycle_cosine(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    L = np.ones(n_epoch)
    period = n_epoch/n_cycle
    step = (stop-start)/(period*ratio) # step is in [0,1]
    
    # transform into [0, pi] for plots: 

    for c in range(n_cycle):

        v , i = start , 0
        while v <= stop:
            L[int(i+c*period)] = 0.5-.5*math.cos(v*math.pi)
            v += step
            i += 1
    return L    