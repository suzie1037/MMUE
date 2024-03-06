import torch
import torch.nn as nn
import torch.nn.functional as F

p=0.2
drop_is_training=True
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)

def passthrough(x, **kwargs):
    return x

def ELUCons(elu, nchan):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)

class InputTransition(nn.Module):
    def __init__(self, outChans, input_Chans=1):
        super(InputTransition, self).__init__()
        self.conv1 = nn.Conv2d(input_Chans, outChans, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(outChans)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(outChans, outChans, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(outChans)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(outChans, outChans, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(outChans)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.relu3(out)
        return out

class DownTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs):
        super(DownTransition, self).__init__()
        self.down_conv = nn.Conv2d(inChans, outChans, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(outChans)
        self.do1 = passthrough
        self.relu1 = nn.ReLU()

        layers = []
        for i in range(nConvs):
            layers += [
                nn.Conv2d(outChans, outChans, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(outChans),
                nn.ReLU()
            ]
        self.stage_block = nn.Sequential(*layers)

        self.relu2 = nn.ReLU()

    def forward(self, x, dropout=False):
        down = self.relu1(self.bn1(self.down_conv(x)))
        down = F.dropout(down, p=p, training=dropout)
        out = self.do1(down)
        out = self.stage_block(out)
        out = self.relu2(torch.add(out, down))
        return out

class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose2d(inChans, outChans // 2, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(outChans // 2)
        self.do1 = passthrough
        self.do2 = nn.Dropout2d()
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

        layers = []
        for i in range(nConvs):
            layers += [
                nn.Conv2d(outChans, outChans, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(outChans),
                nn.ReLU()
            ]
        self.stage_block = nn.Sequential(*layers)

    def forward(self, x, skipx, dropout=False):
        out = self.do1(x)
        skipxdo = self.do2(skipx)
        out = self.relu1(self.bn1(self.up_conv(out)))
        out = F.dropout(out, p=p, training=dropout)
        xcat = torch.cat((out, skipxdo), 1)
        out = self.stage_block(xcat)
        out = self.relu2(torch.add(out, xcat))
        return out


class OutputTransition(nn.Module):
    def __init__(self, inChans):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv2d(inChans, 2, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(2)
        self.conv2 = nn.Conv2d(2, 2, kernel_size=1, stride=1)
        self.relu1 = nn.ReLU()
        self.softmax = F.softmax

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        return out


class UNet(nn.Module):
    def __init__(self, C=16):
        super(UNet, self).__init__()
        self.in_tr = InputTransition(outChans=C)
        self.down_tr32 = DownTransition(inChans=C, outChans=2*C, nConvs=2)
        self.down_tr64= DownTransition(inChans=2*C, outChans=4*C, nConvs=2)
        self.down_tr128 = DownTransition(inChans=4*C, outChans=8*C, nConvs=2)
        self.up_tr128 = UpTransition(inChans=8*C, outChans=8*C, nConvs=2)
        self.up_tr64 = UpTransition(inChans=8*C, outChans=4*C, nConvs=2)
        self.up_tr32 = UpTransition(inChans=4*C, outChans=2*C, nConvs=2)
        self.out_tr = OutputTransition(2*C)


    def forward(self, x, dropout=False):
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16, dropout=dropout)
        out64 = self.down_tr64(out32, dropout=dropout)
        out128 = self.down_tr128(out64, dropout=dropout)
        out = self.up_tr128(out128, out64, dropout=dropout)
        out = self.up_tr64(out, out32, dropout=dropout)
        out = self.up_tr32(out, out16, dropout=dropout)
        return out


class FuseNet(nn.Module):
    def __init__(self, inChans):
        super(FuseNet, self).__init__()
        self.conv1 = nn.Conv2d(inChans, inChans*2, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(inChans*2)
        self.conv2 = nn.Conv2d(inChans*2, 2, kernel_size=1, stride=1)
        self.relu1 = nn.ReLU()
        self.softmax = F.softmax

    def forward(self, x, dropout=False):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        prob = F.softmax(out, dim=1)
        return out, prob

