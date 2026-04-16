import os
import random
from typing import cast

import torch
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import is_image_file
from PIL import Image


class AmazonWebcamDataset(BaseDataset):
    SOURCE_DOMAIN = 'amazon'
    TARGET_DOMAIN = 'webcam'

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
            os.path.join(self.root, self.SOURCE_DOMAIN)
        )
        
        # Load Webcam (target domain B)
        self.webcam_imgs, self.webcam_labels = self._load_domain_images(
            os.path.join(self.root, self.TARGET_DOMAIN)
        )
        
        if len(self.amazon_imgs) == 0:
            raise RuntimeError(f'Found 0 images in {os.path.join(self.root, self.SOURCE_DOMAIN)}')
        if len(self.webcam_imgs) == 0:
            raise RuntimeError(f'Found 0 images in {os.path.join(self.root, self.TARGET_DOMAIN)}')
            
        print(f'Loaded {len(self.amazon_imgs)} Amazon images')
        print(f'Loaded {len(self.webcam_imgs)} Webcam images')

        # Keep preprocessing semantics aligned with global option definitions
        # (`loadSize`, `fineSize`, and `resize_or_crop`).
        self.transform = get_transform(opt)
        
        self.shuffle_indices()

    def _load_image_with_fallback(self, img_path):
        try:
            return Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            fallback_size = max(self.opt.loadSize, self.opt.fineSize)
            return Image.new('RGB', (fallback_size, fallback_size))

    def _build_item_id(self, image_path, label):
        image_id = os.path.splitext(os.path.basename(image_path))[0]
        if image_id.startswith('frame_'):
            image_id = image_id[len('frame_'):]
        return f'{label}_{image_id}'

    @staticmethod
    def _to_grayscale(tensor):
        gray = tensor[0, ...] * 0.299 + tensor[1, ...] * 0.587 + tensor[2, ...] * 0.114
        return gray.unsqueeze(0)

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

        A_pil = self._load_image_with_fallback(amazon_path)
        A_tensor = cast(torch.Tensor, self.transform(A_pil))
        A_path = self._build_item_id(amazon_path, amazon_label)
        
        # Get Webcam sample (B domain)
        webcam_idx = self.webcam_indices[index % len(self.webcam_indices)]
        webcam_path = self.webcam_imgs[webcam_idx]
        webcam_label = self.webcam_labels[webcam_idx]

        B_pil = self._load_image_with_fallback(webcam_path)
        B_tensor = cast(torch.Tensor, self.transform(B_pil))
        B_path = self._build_item_id(webcam_path, webcam_label)

        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if input_nc == 1:  # RGB to gray
            A_tensor = self._to_grayscale(A_tensor)

        if output_nc == 1:  # RGB to gray
            B_tensor = self._to_grayscale(B_tensor)
        
        item = {
            'A': A_tensor,
            'A_paths': A_path,
            'A_label': amazon_label,
            'B': B_tensor,
            'B_paths': B_path,
            'B_label': webcam_label
        }
        
        return item

    def __len__(self):
        return max(len(self.amazon_imgs), len(self.webcam_imgs))
