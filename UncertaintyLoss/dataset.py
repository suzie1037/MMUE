import os
import torch
import scipy.ndimage
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import transform

DATAROOT = "./PETCTdataset"

CT_MEAN = -397.34
CT_STD = 446.90

PET_MEAN = 65.42
PET_STD = 124.84

train_transform = transform.Compose([
    transform.RandScale([0.9, 1.1]),
    transform.RandRotate([-10, 10], padding=[0], ignore_label=0),
    transform.RandomHorizontalFlip(),
    transform.Crop([96, 128], crop_type='rand', padding=[0], ignore_label=0),
    transform.ToTensor()
])

val_transform = transform.Compose([transform.Resize([96, 128]), transform.ToTensor()])

def normalize(img):
    max_ = np.max(np.max(np.max(img)))
    min_ = np.min(np.min(np.min(img)))
    img = (img - min_) / (max_ - min_)
    img = (img * 255.).astype(np.uint8)
    return img


### order=1: bilinear order=0: nearest
def img_resize(image, resize, order=1):
    img_size = image.shape
    zoom = (resize[0] / img_size[0], resize[1] / img_size[1], resize[2] / img_size[2])
    if resize[0] == img_size[0] and resize[1] == img_size[1]:
        return image
    image = scipy.ndimage.interpolation.zoom(image, zoom=zoom, order=order)
    return image

def read_data(data_dir, fold="fold1"):
    pet_train = np.load(os.path.join(data_dir, fold, "pet_train.npy"))
    ct_train = np.load(os.path.join(data_dir, fold, "ct_train.npy"))
    label_train = np.load(os.path.join(data_dir, fold, "label_train.npy"))

    pet_val = np.load(os.path.join(data_dir, fold, "pet_val.npy"))
    ct_val = np.load(os.path.join(data_dir, fold, "ct_val.npy"))
    label_val = np.load(os.path.join(data_dir, fold, "label_val.npy"))

    pet_test = np.load(os.path.join(data_dir, "test_pet_453.npy"))
    ct_test = np.load(os.path.join(data_dir, "test_ct_453.npy"))
    label_test = np.load(os.path.join(data_dir, "test_label_453.npy"))

    train_dict = {"pet": pet_train, "ct": ct_train, "label": label_train}
    val_dict = {"pet": pet_val, "ct": ct_val, "label": label_val}
    test_dict = {"pet": pet_test, "ct": ct_test, "label": label_test}
    return train_dict, val_dict, test_dict

class SegDataset(torch.utils.data.Dataset):
    def __init__(self, data_dict, transform):
        self.pet = data_dict["pet"]
        self.ct = data_dict["ct"]
        self.label = data_dict["label"]
        self.pet = (self.pet - PET_MEAN) / PET_STD
        self.ct = (self.ct - CT_MEAN) / CT_STD
        # self.ct[self.ct < -240] = -240
        # self.ct[self.ct > 160] = 160
        self.transform = transform

    def __getitem__(self, idx):
        pet, ct, label = self.transform(self.pet[:, :, idx], self.ct[:, :, idx], self.label[:, :, idx])
        return pet, ct, label

    def __len__(self):
        return self.label.shape[-1]


if __name__ == "__main__":

    #########  Dataset  ##########
    train_dict, val_dict, test_dict = read_data(DATAROOT)
    train_set = SegDataset(train_dict, train_transform)
    test_set = SegDataset(test_dict, val_transform)

    dataloader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
    print(len(dataloader))
    for i, batch in enumerate(dataloader):
        print(i)
        if(i%50==0):
            pet_image, ct_image, mask = batch
            fig = plt.figure()
            a = fig.add_subplot(1,3,1)
            plt.imshow(pet_image[0][0], cmap='hot')
            a = fig.add_subplot(1,3,2)
            plt.imshow(ct_image[0][0], cmap='gray')
            a = fig.add_subplot(1,3,3)
            plt.imshow(mask[0], cmap='gray')
            plt.show()
            print('process finished')