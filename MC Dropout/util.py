import torch

def dice_score(output, target):
    output = torch.squeeze(output)
    target = torch.squeeze(target)
    intersection = torch.sum(output*target).float().data
    l = torch.sum(output).float().data
    r = torch.sum(target).float().data
    dice = (2 * intersection + 1e-5) / (l + r + 1e-5)
    return dice

def CE_VE(g, m):
    g = torch.squeeze(g)
    m = torch.squeeze(m)
    g_positive = (g == 1)
    m_positive = (m == 1)
    g_negative = (g == 0)
    m_negative = (m == 0)

    m_FalseNegative = torch.sum(m_negative * g_positive)
    m_FalsePositive = torch.sum(m_positive * g_negative)

    cm = torch.sum(m_positive).float()
    cg = torch.sum(g_positive).float()
    CE = (((torch.abs(m_FalseNegative) + torch.abs(m_FalsePositive))+1e-5).float() /
          (max(cg+1e-5, (torch.abs(m_FalseNegative) + torch.abs(m_FalsePositive))+1e-5))).data
    VE = ((torch.abs(cg - cm)+1e-5).float() /
          (max(cg+1e-5, torch.abs(cg - cm)+1e-5))).data
    return CE, VE
