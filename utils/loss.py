import torch
import torch.nn as nn


class CoordLoss(nn.Module):
    def __init__(self, method, dim):
        super(CoordLoss, self).__init__()
        if method == "L1":
            self.ftn = torch.abs
        if method == "MSE":
            self.ftn = torch.square

        if method == "L1 + MSE":
            self.ftn = self.dual
        self.dim = dim        

    def dual(self, x):
        return (torch.abs(x) + torch.square(x)) / 2


    def forward(self, coord_out, coord_gt, valid=None, is_3D=None):
        if valid is None:
            loss = self.ftn(coord_out - coord_gt)
        else:
            loss = self.ftn(coord_out - coord_gt) * valid
        
        if is_3D is not None:
            loss_z = loss[:,:,2:] * is_3D[:,None,None].float()
            loss = torch.cat((loss[:,:,:2], loss_z),2)

        if self.dim == "msm":
            loss = loss.sum(dim=1)
            loss = loss.mean()
        
        elif self.dim == "mss":
            loss = loss.mean(dim=0)
            loss = loss.sum()

        return loss


class ParamLoss(nn.Module):
    def __init__(self, method, dim):
        super(ParamLoss, self).__init__()
        if method == "L1":
            self.ftn = torch.abs
        if method == "MSE":
            self.ftn = torch.square
        self.dim = dim        

    def forward(self, param_out, param_gt, valid=None):
        if valid is None:
            loss = self.ftn(param_out - param_gt)
        else:
            loss = self.ftn(param_out - param_gt) * valid

        if self.dim == "msm":
            loss = loss.sum(dim=1)
            loss = loss.mean()
        
        elif self.dim == "mss":
            loss = loss.mean(dim=0)
            loss = loss.sum()
        return loss


class KLLoss(nn.Module):
    def __init__(self, cfg):
        super(KLLoss, self).__init__()
        self.batch_size = cfg.batch_size / cfg.num_gpus

    def forward(self, mean1, logvar1, mean2, logvar2):
        sigma1 = logvar1.mul(0.5).exp()
        sigma2 = logvar2.mul(0.5).exp()
        kld = torch.log(sigma2/sigma1) + (torch.exp(logvar1) + (mean1-mean2)**2)/(2*torch.exp(logvar2)) - 1/2
        return kld.sum() / self.batch_size


class AccelLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, mesh, valid_mask):
        velocity = mesh[:, 1:, :, :] - mesh[:, :-1, :, :]
        accel = velocity[:, 1:, :, :] - velocity[:, :-1, :, :]

        accel_mask = valid_mask[:, 2:].unsqueeze(2).unsqueeze(3)
        accel_loss = torch.square(accel) * accel_mask

        return accel_loss
