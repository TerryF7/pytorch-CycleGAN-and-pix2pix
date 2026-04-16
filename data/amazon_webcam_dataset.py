import os
import random

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset
from data.image_folder import is_image_file
from PIL import Image

RESAMPLE_BICUBIC = getattr(getattr(Image, 'Resampling', Image), 'BICUBIC')


class AmazonWebcamDataset(BaseDataset):
    """
    Load Amazon and Webcam domain adaptation dataset.
    Structure expected:
    dataroot/
      amazon/
        class_0/
          img1.jpg
        class_1/
          ...
      webcam/
        class_0/
          img1.jpg
        ...
    """
    
    def name(self):
        return 'AmazonWebcamDataset'

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        
        # Load Amazon (source domain A)
        self.amazon_imgs, self.amazon_labels = self._load_domain_images(
            os.path.join(self.root, 'amazon')
        )
        
        # Load Webcam (target domain B)
        self.webcam_imgs, self.webcam_labels = self._load_domain_images(
            os.path.join(self.root, 'webcam')
        )
        
        if len(self.amazon_imgs) == 0:
            raise RuntimeError(f'Found 0 images in {os.path.join(self.root, "amazon")}')
        if len(self.webcam_imgs) == 0:
            raise RuntimeError(f'Found 0 images in {os.path.join(self.root, "webcam")}')
            
        print(f'Loaded {len(self.amazon_imgs)} Amazon images')
        print(f'Loaded {len(self.webcam_imgs)} Webcam images')

        # Office-31 has mixed image sizes. Force a fixed spatial size so
        # DataLoader can collate tensors into a batch safely.
        if opt.isTrain and not opt.no_flip:
            self.transform = transforms.Compose([
                transforms.Resize((opt.fineSize, opt.fineSize), RESAMPLE_BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((opt.fineSize, opt.fineSize), RESAMPLE_BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        
        self.shuffle_indices()

    def _load_domain_images(self, domain_path):
        """Load images from domain directory organized by class folders."""
        imgs = []
        labels = []
        
        if not os.path.exists(domain_path):
            print(f"Warning: Domain path {domain_path} does not exist")
            return imgs, labels
        
        # Iterate through class folders
        for class_idx, class_name in enumerate(sorted(os.listdir(domain_path))):
            class_path = os.path.join(domain_path, class_name)
            if not os.path.isdir(class_path):
                continue
                
            # Collect all image files in this class
            for img_name in sorted(os.listdir(class_path)):
                if is_image_file(img_name):
                    img_path = os.path.join(class_path, img_name)
                    imgs.append(img_path)
                    labels.append(class_idx)
        
        return imgs, labels

    def shuffle_indices(self):
        self.amazon_indices = list(range(len(self.amazon_imgs)))
        self.webcam_indices = list(range(len(self.webcam_imgs)))
        
        if not self.opt.serial_batches:
            random.shuffle(self.amazon_indices)
            random.shuffle(self.webcam_indices)

    def __getitem__(self, index):
        if index == 0:
            self.shuffle_indices()
        
        # Get Amazon sample (A domain)
        amazon_idx = self.amazon_indices[index % len(self.amazon_indices)]
        amazon_path = self.amazon_imgs[amazon_idx]
        amazon_label = self.amazon_labels[amazon_idx]
        
        try:
            A_img = Image.open(amazon_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {amazon_path}: {e}")
            # Return a dummy image on error
            A_img = Image.new('RGB', (256, 256))
        
        A_img = self.transform(A_img)
        A_path = os.path.basename(amazon_path)
        
        # Get Webcam sample (B domain)
        webcam_idx = self.webcam_indices[index % len(self.webcam_indices)]
        webcam_path = self.webcam_imgs[webcam_idx]
        webcam_label = self.webcam_labels[webcam_idx]
        
        try:
            B_img = Image.open(webcam_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {webcam_path}: {e}")
            # Return a dummy image on error
            B_img = Image.new('RGB', (256, 256))
        
        B_img = self.transform(B_img)
        B_path = os.path.basename(webcam_path)
        
        item = {
            'A': A_img,
            'A_paths': A_path,
            'A_label': amazon_label,
            'B': B_img,
            'B_paths': B_path,
            'B_label': webcam_label
        }
        
        return item

    def __len__(self):
        return max(len(self.amazon_imgs), len(self.webcam_imgs))
