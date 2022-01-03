import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
# from data_aug.gaussian_blur import GaussianBlur
import data_aug.transformers_multi as transforms_new
from torchvision import datasets
import pandas as pd
from PIL import Image
from skimage import io, img_as_ubyte
import torch
import os
import cv2
from random import choice

np.random.seed(0)


class Dataset():
    def __init__(self, csv_file, transform=None, input_c=5, data_root="/mnt/yifan/data/blackgrass",
                 list_dir="/mnt/yifan/data/blackgrass/blackgrass/data_table.txt"):
        self.files_list = pd.read_csv(csv_file)
        self.lists = []
<<<<<<< HEAD
        self.root = "/home/fg405/rds/hpc-work/IMAGE_ARCHIVE/single/"
        for path in open("/home/fg405/rds/hpc-work/IMAGE_ARCHIVE/data_table_early_field23.txt"):
=======
        self.input_c = input_c
        self.root = data_root + "/single/"
        for path in open(list_dir):
>>>>>>> 2494109316606d8d0502dc55d9c7cd6aa27931c8
            if "tif" not in path:
                continue
            else:
                res = path.split()
                r_prefix = res[0].split("/")[-1]  # Red
                g_prefix = res[1].split("/")[-1]  # green
                b_prefix = res[2].split("/")[-1]  # blue
                e_prefix = res[3].split("/")[-1]  # red edge
                nir_prefix = res[4].split("/")[-1]  # NIR
                class_label = res[6]
                path_ids = os.listdir(self.root + class_label + '/' + r_prefix.replace(".tif", ""))
                for path_id in path_ids:
                    self.lists.append(
                        r_prefix + "\t" + g_prefix + "\t" + b_prefix + "\t" + e_prefix + "\t" + nir_prefix + "\t" + path_id + "\t" + class_label)

        self.transform = transform

    def __len__(self):
        return len(self.lists)

    def __getitem__(self, idx):
        temp_path = self.lists[idx].split()
        # path_id = "/" + choice(os.listdir(self.root + temp_path[-1] + '/' + temp_path[0].replace(".tif", "")))
        input_channel = self.input_c
        img = cv2.imread(self.root + temp_path[-1] + '/' + temp_path[0].replace(".tif", "") + "/" + temp_path[-2])[:, :,
              0:1]
        img = transforms.functional.to_tensor(img)
        for ind in range(input_channel - 1):
            img_new = cv2.imread(
                self.root + temp_path[-1] + '/' + temp_path[ind + 1].replace(".tif", "") + "/" + temp_path[-2])[:,
                      :, 0:1]
            img_new = transforms.functional.to_tensor(img_new)
            img = torch.cat((img, img_new), 0)
        # img_b = cv2.imread(self.root + temp_path[-1] + '/' + temp_path[2].replace(".tif", "") + path_id)
        # img_e = cv2.imread(self.root + temp_path[-1] + '/' + temp_path[3].replace(".tif", "") + path_id)
        # img_n = cv2.imread(self.root + temp_path[-1] + '/' + temp_path[4].replace(".tif", "") + path_id)
        # img = transforms.functional.to_tensor(img_r)
        # img2 = torch.cat((img, img), 0)
        if self.transform:
            sample = self.transform(img)
        return sample


#
# class ToPIL(object):
#     def __call__(self, sample):
#         img = sample
#         img = transforms.functional.to_pil_image(img)
#         return img


class DataSetWrapper(object):

    def __init__(self, batch_size, num_workers, valid_size, input_shape, s, input_c, root_dir, list_dir):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.s = s
        self.input_c = input_c
        self.input_shape = eval(input_shape)
        self.root = root_dir
        self.lists = list_dir

    def get_data_loaders(self):
        data_augment = self._get_simclr_pipeline_transform()
        train_dataset = Dataset(csv_file='all_patches.csv', transform=SimCLRDataTransform(data_augment),
                                input_c=self.input_c, data_root=self.root, list_dir=self.lists)
        train_loader, valid_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader

    def _get_simclr_pipeline_transform(self):
        # get a set of data augmentation transformations as described in the SimCLR paper.
        color_jitter = transforms_new.ColorJitter(0.8 * self.s, 0.8 * self.s, 0.8 * self.s, 0.2 * self.s)
        data_transforms = transforms_new.Compose([transforms_new.ToPIL(),
                                                  transforms_new.RandomResizedCrop(size=self.input_shape[0]),
                                                  transforms_new.RandomHorizontalFlip(),
                                                  transforms_new.RandomApply([color_jitter], p=0.8),
                                                  transforms_new.RandomGrayscale(p=0.2),
                                                  transforms_new.GaussianBlur(
                                                      kernel_size=int(0.06 * self.input_shape[0])),
                                                  transforms_new.ToTensor()])
        return data_transforms

    def get_train_validation_data_loaders(self, train_dataset):
        # obtain training indices that will be used for validation
        num_train = len(train_dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)

        split = int(np.floor(self.valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, drop_last=True, shuffle=False)
        valid_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                  num_workers=self.num_workers, drop_last=True)
        return train_loader, valid_loader


class SimCLRDataTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        xi1, xi2 = self.transform(sample[0:3], sample[-4:-1])
        xi = torch.rand_like(sample)
        xi[0:3] = xi1
        xi[-4:-1] = xi2
        xj1, xj2 = self.transform(sample[0:3], sample[-4:-1])
        xj = torch.rand_like(sample)
        xi[0:3] = xj1
        xi[-4:-1] = xj2
        return xi, xj
