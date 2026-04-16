import os.path
import random

import numpy as np
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset
from PIL import Image
from torchvision.datasets.mnist import MNIST
from torchvision.datasets.usps import USPS


class MnistUspsDataset(BaseDataset):
    def name(self):
        return 'MnistUspsDataset'

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        print(opt)
        
        # Load MNIST (28x28 grayscale)
        self.mnist = MNIST(os.path.join(opt.dataroot, 'mnist'),
                           train=opt.isTrain, download=True)
        
        # Load USPS (16x16 grayscale)
        self.usps = USPS(os.path.join(opt.dataroot, 'usps'),
                         train=opt.isTrain, download=True)

        # Transform: normalize to [-1, 1] like CycleGAN convention
        # For grayscale images: single channel, so normalize with single mean/std
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])
        
        # Lambda to convert grayscale (1, H, W) to RGB (3, H, W)
        self.to_rgb = transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x)

        self.shuffle_indices()

    def shuffle_indices(self):
        self.mnist_indices = list(range(len(self.mnist)))
        self.usps_indices = list(range(len(self.usps)))
        print('num mnist', len(self.mnist_indices), 'num usps', len(self.usps_indices))
        if not self.opt.serial_batches:
            random.shuffle(self.mnist_indices)
            random.shuffle(self.usps_indices)

    def __getitem__(self, index):
        if index == 0:
            self.shuffle_indices()

        # Get MNIST sample (A domain)
        A_img, A_label = self.mnist[self.mnist_indices[index % len(self.mnist)]]
        # MNIST is 28x28, resize to 32x32 to match USPS (will be resized to 32x32)
        A_img = A_img.resize((32, 32))
        A_img = self.transform(A_img)  # Shape: (1, 32, 32)
        A_img = self.to_rgb(A_img)  # Convert to (3, 32, 32)
        A_path = '%01d_%05d.png' % (A_label, index)

        # Get USPS sample (B domain)
        B_img, B_label = self.usps[self.usps_indices[index % len(self.usps)]]
        # USPS is 16x16, resize to 32x32 for consistency
        B_img = B_img.resize((32, 32))
        B_img = self.transform(B_img)  # Shape: (1, 32, 32)
        B_img = self.to_rgb(B_img)  # Convert to (3, 32, 32)
        B_path = '%01d_%05d.png' % (B_label, index)

        item = {}
        item.update({'A': A_img,
                     'A_paths': A_path,
                     'A_label': A_label
                 })
        
        item.update({'B': B_img,
                     'B_paths': B_path,
                     'B_label': B_label
                 })
        return item
        
    def __len__(self):
        return max(len(self.mnist), len(self.usps))
