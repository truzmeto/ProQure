import torch
import torch.nn as nn

class WeMSE(torch.nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.w = weights
        
    def forward(self, x, y):
        loss = self.w * (y - x).pow(2).mean(dim=(2,3,4)).sum(0)         
        return loss.sum()

    

class VAELoss(torch.nn.Module):
    def __init__(self, fa=1.0, fb=0.1):
        super().__init__()

        self.a = fa
        self.b = fb
        self.log_scale = nn.Parameter(torch.Tensor([0.0])).cuda()
        #self.mseL = nn.MSELoss()

            
    def ReconL(self, out, inp, scale):
        dist = torch.distributions.Normal(out, scale) 
        log_pxz = dist.log_prob(inp)
                 
        return log_pxz.mean(dim=(1, 2, 3, 4))


    def latent_loss(self, mu, std):
        mu_sq = mu * mu
        std_sq = std * std
        #return 0.5 * torch.mean(mu_sq + std_sq - torch.log(std_sq) - 1.0, dim=1)
        return 0.5 * torch.mean(mu_sq + std_sq - 1.0, dim=1)


    def KLdivL(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        
        #z = z.squeeze(); mu = mu.squeeze(); std = std.squeeze()
        
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)
        #kl-div
        kl =  log_qzx - log_pz 
        kl = kl.sum(1)
        #print(kl.sum())
        return kl*2/(32*32*32)     
   
    
    def forward(self, out, inp, z, mu, std):
        
        scale = torch.exp(self.log_scale) 
        recon_loss = self.ReconL(out, inp, scale)
        #kl_loss = self.KLdivL(z, mu, std)
        #loss = self.b*kl_loss - self.a*recon_loss #+ 50.0*self.mseL(out, inp)
        #loss = self.b*self.latent_loss(mu, std) - self.a*recon_loss #+ 50.0*self.mseL(out, inp)
        loss = - self.a*recon_loss #+ 50.0*self.mseL(out, inp)

        return loss.sum() #batch sum






class CLoss(torch.nn.Module):
    def __init__(self, fa=1.0, fb=1.0):
        super().__init__()

        self.a = fa
        self.b = fb
        self.RegL = nn.MSELoss()
        self.SegL = nn.BCELoss() #reduction="batchmean")
   
         
    def forward(self, inp, outV, outU, OneHot):

        #outU = outU.softmax(dim=1)
        loss = self.a*self.RegL(outV, inp) + self.b*self.SegL(outU, OneHot)  
        
        return loss



class XL1Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, output, target):

        L1 = nn.L1Loss(reduction='none')
        loss = L1(output,target) * target #too unstable!
        return loss.sum()

    
class XL2Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, output, target):

        L2 = nn.MSELoss(reduction='none')
        loss = L2(output,target) * target #too unstable
        return loss.sum()

    
class PWLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, output, target):
        """
        Note, reduction mean results in very small loss
        """
        
        T = torch.clamp(target, min = 0.2, max = 1.0)# target.max())
        W = T / T.sum()
        
        L1 = nn.L1Loss(reduction='none')
        loss = L1(output,target) * W

        return loss.sum()
    

class GFE_WLoss(torch.nn.Module):
    def __init__(self): #??????????????????????????
        super().__init__()
    
    def forward(self, output, target, min_val=-2.0, max_val=2.0):

        T = torch.clamp(target, min = min_val, max = max_val)
        W = -T / T.abs().sum()
        
        L1 = nn.L1Loss(reduction='none')
        loss = L1(output,target) * W

        return loss.mean()

    
class WLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, output, target, thresh = 0.07):

        mask1 = target > thresh
        l1 = (output[mask1] - target[mask1]).pow(2).mean().sqrt()

        mask2 = target > thresh
        l2 = (output[mask2] - target[mask2]).pow(2).mean().sqrt()

        loss = 0.8*l1 + 0.2*l2
        
        return loss



class logCoshLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        diff = output - target
        return torch.mean(torch.log(torch.cosh(diff + 1e-12)))



class XTanhLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()



    def forward(self, output, target):
        diff = output - target
        return torch.mean(diff * torch.tanh(diff))



class XSigmoidLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        diff = output - target
        return torch.mean(2 * diff / (1 + torch.exp(-diff)) - diff)

    
class Regularization(torch.nn.Module):
    """
    Class calculates L1 or L2 regularization terms
    by accessing model parameters(weights).    
    """

    def __init__(self):
        super().__init__()
    
    def forward(self, model, RegType="L1", Lambda=0.00001):

        if RegType == "L1":
            L1 = 0.0
            for w in model.parameters():
                L1 = L1 + w.norm(1)
            L = L1*Lambda
        
        if RegType == "L2":
            L2 = 0.0
            for w in model.parameters():
                L2 = L2 + w.norm(2)
            L = L2*Lambda

        return L
    
    
class PenLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

        
    def StepF(self, inp, thresh):
        return 0.5*(1.0 - torch.sign(inp + thresh))
    
    
    def forward(self, output, target, thresh):
        
        loss = torch.abs(output - target) * \
               self.StepF(output + target, -2.0*thresh) - \
               2.0*(output - thresh) * \
               self.StepF(-output - target, 2.0*thresh) * \
               self.StepF(-target, thresh) * \
               self.StepF(output, -thresh) - \
               2.0*(target - thresh) * \
               self.StepF(-output - target, 2.0*thresh) * \
               self.StepF(-output, thresh) * \
               self.StepF( target, -thresh)
        
        
        return loss.mean()



class BinLoss(torch.nn.Module):
    """
    Does not work!
    """
    def __init__(self):
        super(BinLoss,self).__init__()


    def BinData(self, inp, Blist):

        batch_size, nchannel, nx, ny, nz = inp.shape
        nvox = nx*ny*nz

        out = torch.empty(batch_size, nchannel, len(Blist))

        for ibatch in range(batch_size):
            for ichan in range(nchannel):

                inp1 = inp[ibatch, ichan,:,:,:].squeeze()
                freqs = torch.tensor([len(inp1[(inp1 >= el[0]) & (inp1 < el[1])]) for el in Blist],
                                     dtype = torch.float32)
                #, requires_grad=True)

                out[ibatch,ichan,:] = freqs / nvox

        return out


    def forward(self, inp, tar, bin_range):

        sl1 = nn.SmoothL1Loss()
        loss1 = sl1(inp,tar)
        loss2 = torch.abs(self.BinData(inp, bin_range)
                          - self.BinData(tar, bin_range)).mean()

        #loss2 = torch.tensor(loss2, requires_grad=True)
        return  0.5*loss1 + 0.5*loss2






