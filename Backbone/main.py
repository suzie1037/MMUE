import os
import torch
import argparse
from torch.utils.data import DataLoader
from dataset import SegDataset, read_data
import transform
from model import CoSeg
from unet import weights_init
from trainer import train
# Arguments
parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('--fold', default='fold1')
parser.add_argument('--data_root', default='./PETCTdataset')
parser.add_argument('--model_name', default='baseline')
parser.add_argument('--save_dir', default='./SavedModel/fold1')
parser.add_argument('--epoch_num', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--class_num' , type=int, default=2)
parser.add_argument('--learning_rate', type=float, default=0.001, help='initial learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
parser.add_argument('--optimizer', default='Adam', help='Adam or SGD')
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

if __name__ == "__main__":
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    CUDA = args.gpu is not None
    GPU_ID = args.gpu

    train_transform = transform.Compose([
        transform.RandScale([0.9, 1.1]),
        transform.Crop([96, 128], crop_type='rand', padding=[0], ignore_label=0),
        transform.ToTensor()
    ])
    val_transform = transform.Compose([transform.Resize((96, 128)), transform.ToTensor()])

    #########  Dataset  ##########
    train_dict, val_dict, test_dict = read_data(args.data_root, args.fold)
    train_set = SegDataset(train_dict, train_transform)
    val_set = SegDataset(val_dict, val_transform)
    test_set = SegDataset(test_dict, val_transform)
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=16)
    val_dataloader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=16)
    test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=16)
    dataloader_dict = {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}

    seg_model = CoSeg().cuda(GPU_ID)
    seg_model.apply(weights_init)  ## kaiming_normal

    Adam_optimizer = torch.optim.Adam([{'params': seg_model.parameters()}],
                                      weight_decay = args.weight_decay,
                                     lr=args.learning_rate
                                 )
    train(args, dataloader_dict, args.epoch_num, seg_model, Adam_optimizer)