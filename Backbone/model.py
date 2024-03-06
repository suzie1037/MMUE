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
    def get_loss(self, label):
        faltten_label = label.view(label.numel())
        flatten_output = self.output.permute(0, 2, 3, 1).contiguous()
        flatten_output = flatten_output.view(flatten_output.numel() // 2, 2)
        output_loss = F.cross_entropy(flatten_output, faltten_label)
        return output_loss
