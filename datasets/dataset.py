import os
import torch
import torch.nn as nn
import torch.utils.data as data
import cv2
import numpy as np
from tqdm import tqdm


class FacemaskDataset(data.Dataset):
    def __init__(self, cfg):
        self.root_dir = cfg.root_dir
        self.cfg = cfg

        self.mask_folder = os.path.join(self.root_dir, 'image_binary')
        self.img_folder = os.path.join(self.root_dir, 'image_masked')
        self.ground_truth_folder = os.path.join(self.root_dir, 'image')
        self.load_images()
        
    def load_images(self):
        self.fns = []
        idx = 0
        ground_truth_paths = sorted(os.listdir(self.ground_truth_folder))
        for ground_truth_name in ground_truth_paths:
            mask_name = ground_truth_name.split('.')[0]+'_binary.jpg'
            img_name = ground_truth_name.split('.')[0]+'_masked.jpg'
            img_path = os.path.join(self.img_folder, img_name)
            mask_path = os.path.join(self.mask_folder, mask_name)
            ground_truth_path = os.path.join(self.ground_truth_folder, ground_truth_name)
            if os.path.isfile(mask_path): 
                self.fns.append([img_path, mask_path, ground_truth_path])

    def __getitem__(self, index):
        img_path, mask_path, ground_truth_path = self.fns[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.cfg.img_size, self.cfg.img_size))

        ground_truth = cv2.imread(ground_truth_path)
        ground_truth = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2RGB)
        ground_truth = cv2.resize(ground_truth, (self.cfg.img_size, self.cfg.img_size))
        
        
        mask = cv2.imread(mask_path, 0)
        
        mask[mask>0]=1.0
        mask = np.expand_dims(mask, axis=0)
    
        img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        mask = torch.from_numpy(mask.astype(np.float32)).contiguous()
        ground_truth = torch.from_numpy(ground_truth.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        return img, mask, ground_truth
    
    def collate_fn(self, batch):
        imgs = torch.stack([i[0] for i in batch])
        masks = torch.stack([i[1] for i in batch])
        ground_truth = torch.stack([i[2] for i in batch])
        return {
            'imgs': imgs,
            'masks': masks,
            'ground_truth' : ground_truth
        }
    
    def __len__(self):
        return len(self.fns)
