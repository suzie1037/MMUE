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

    def forward(self, pet, ct, dropout=False):
        pet_feature = self.pet_branch(pet, dropout=dropout)
        ct_feature = self.ct_branch(ct, dropout=dropout)
        fuse_feature = torch.cat((pet_feature, ct_feature), 1)
        self.output, self.prob = self.fuse(fuse_feature, dropout=dropout)
        return self.output, self.prob

    ### use cross_entropy loss
    def get_loss(self, output, label, uncertainty, theta):
        prob = F.softmax(output, dim=1)
        ce_map = -torch.log(prob[:, 1, :, :] + 1e-5) * ((label == 1).int()) - torch.log(prob[:, 0, :, :] + 1e-5) * (
            (label == 0).int())
        score = torch.mean(ce_map * (theta*uncertainty + 1.0))
        return score

