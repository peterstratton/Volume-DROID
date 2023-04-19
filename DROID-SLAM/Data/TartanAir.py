"""
Custom torch.utils.data.Dataset for TartanAir data.
"""
import os
import numpy as np
import yaml
from torch.utils.data import Dataset
import torch
import math
from scipy.spatial.transform import Rotation as R
from PIL import Image
import glob
import cv2


#TODO: Make a yaml file?
config_file = os.path.join('Config/tartanair.yaml')
tartanair_config = yaml.safe_load(open(config_file, 'r'))
SPLIT_SEQUENCES = tartanair_config["SPLIT_SEQUENCES"]

class TartanAirDataset(Dataset):
    """
    TartanAir Dataset for Neural BKI project
    """
    def __init__(self,
                directory="/datasets_withaccess/POO2", #change to dataset directory later in DROID-SLAM
                device='cuda',
                image_size=[384, 512], 
                intrinsics_vec=[320.0, 320.0, 320.0, 240.0],
                 ):
        
        classes_from = [217, 152, 151, 205, 237, 224, 251, 227, 168, 196, 146]
        classes_to = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        # read all png images in folder
        ht0, wd0 = [480, 640]
        images_left = sorted(glob.glob(os.path.join(directory, 'image_left/*.png')))
        images_right = sorted(glob.glob(os.path.join(directory, 'image_right/*.png')))
        depth_left = sorted(glob.glob(os.path.join(directory, 'depth_left/*.npy')))
        depth_right = sorted(glob.glob(os.path.join(directory, 'depth_right/*.npy')))
        seg_left = sorted(glob.glob(os.path.join(directory, 'seg_left/*.npy')))
        seg_right = sorted(glob.glob(os.path.join(directory, 'seg_right/*.npy')))

        self.data = []
        for t in range(len(images_left)):
            # read RGB images 
            images = [cv2.resize(cv2.imread(images_left[t]), (image_size[1], image_size[0])), 
                   cv2.resize(cv2.imread(images_right[t]), (image_size[1], image_size[0]))]
            images = torch.from_numpy(np.stack(images, 0)).permute(0,3,1,2)

            # read corresponding depths
            depths = [np.load(depth_left[t]), np.load(depth_right[t])]
            depths = torch.from_numpy(np.stack(depths, 0))

            # read corresponding segs
            seg_l = np.load(seg_left[t])
            seg_r = np.load(seg_right[t])

            # hack to match TartanAir classes to KITTI classes, classes outside the 
            # set are set to label 0
            for cl_f, cl_t in zip(classes_from, classes_to):
                seg_l = np.where(seg_l == cl_f, cl_t, seg_l)
                seg_r = np.where(seg_r == cl_f, cl_t, seg_r)
            seg_l = np.where(seg_l > 10, 0, seg_l)
            seg_r = np.where(seg_r > 10, 0, seg_r)
            segs = [seg_l, seg_r]

            segs = torch.from_numpy(np.stack(segs, 0))

            intrinsics = .8 * torch.as_tensor(intrinsics_vec)
            self.data.append((t, images, depths, segs, intrinsics))

    def collate_fn(self, data):
        t_batch = [d[0] for d in data]
        imgs_batch = [d[1] for d in data]
        dps_batch = [d[2] for d in data]
        segs_batch = [d[3] for d in data]
        intrinsics_batch = [d[4] for d in data]
        return t_batch, imgs_batch, dps_batch, segs_batch, intrinsics_batch

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]