import torch
import torch.nn as nn
import torch.nn.functional as F
from unet import UNet, FuseNet

class CoSeg(nn.Module):
    def __init__(self, C=16):
        super(CoSeg, self).__init__()
        self.pet_branch = UNet(C=C)
        self.ct_branch = UNet(C=C)
        self.fuse = FuseNet(inChans=4*C)
        self.stdNet = FuseNet(inChans=4*C)

    def forward(self, pet, ct, dropout=False):
        pet_feature = self.pet_branch(pet)
        ct_feature = self.ct_branch(ct)
        fuse_feature = torch.cat((pet_feature, ct_feature), 1)
        self.raw_output, _ = self.fuse(fuse_feature)
        self.std, _ = self.stdNet(fuse_feature)
        self.std = torch.log(1 + torch.exp(self.std)) + 1e-6
        if dropout:
            eplison = torch.randn_like(self.std)
        else:
            eplison = torch.zeros_like(self.std)
        self.output = self.raw_output + eplison * self.std
        self.prob = F.softmax(self.output, dim=1)

        return self.output, self.prob

    ### use cross_entropy loss
    def get_loss(self, label):
        eplison1 = torch.randn_like(self.std)
        eplison2 = torch.randn_like(self.std)
        eplison3 = torch.randn_like(self.std)
        eplison4 = torch.randn_like(self.std)
        eplison5 = torch.randn_like(self.std)
        prob = (F.softmax(self.raw_output + eplison1 * self.std, dim=1) + F.softmax(self.raw_output + eplison2 * self.std, dim=1)+
                F.softmax(self.raw_output + eplison3 * self.std, dim=1) + F.softmax(self.raw_output + eplison4 * self.std, dim=1) +
                F.softmax(self.raw_output + eplison5 * self.std, dim=1))/5
        ce_map = -torch.log(prob[:, 1, :, :]) * ((label == 1).int()) - torch.log(prob[:, 0, :, :]) * (
            (label == 0).int())
        score = torch.mean(ce_map)
        return score


