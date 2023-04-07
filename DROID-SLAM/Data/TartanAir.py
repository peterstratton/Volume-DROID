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
    """Kitti Dataset for Neural BKI project
    
    Access to the processed data, including evaluation labels predictions velodyne poses times
    """
    """
    Initialize the TartanAirDataset object with the following parameters:

    grid_params: dictionary containing the following parameters
        - grid_size: size of the voxel grid
        - min_bound: minimum XYZ coordinates for the voxel grid
        - max_bound: maximum XYZ coordinates for the voxel grid
    directory: path to the directory containing the KITTI dataset
    device: device on which to perform computations
    num_frames: number of consecutive frames to use as input
    voxelize_input: whether to voxelize the input point cloud
    binary_counts: whether to use binary voxel counts instead of occupancy probabilities
    random_flips: whether to randomly flip the input data during training
    use_aug: whether to use data augmentation during training
    apply_transform: whether to apply a transform matrix to the input point cloud
    remap: whether to remap the class labels to a new range
    from_continuous: whether to load predictions from a continuous model
    to_continuous: whether to convert predictions to a continuous output
    num_classes: number of classes in the dataset
    data_split: which split of the dataset to use (train, val, or test)
    """
    def __init__(self,
                directory="/datasets_withaccess/POO2", #change to dataset directory later in DROID-SLAM
                device='cuda',
                image_size=[384, 512], 
                intrinsics_vec=[320.0, 320.0, 320.0, 240.0],
                 ):
        
        # read all png images in folder
        ht0, wd0 = [480, 640]
        images_left = sorted(glob.glob(os.path.join(directory, 'image_left/*.png')))
        images_right = sorted(glob.glob(os.path.join(directory, 'image_right/*.png')))
        depth_left = sorted(glob.glob(os.path.join(directory, 'depth_left/*.npy')))
        depth_right = sorted(glob.glob(os.path.join(directory, 'depth_right/*.npy')))

        self.data = []
        for t in range(len(images_left)):
            # read RGB images 
            images = [cv2.resize(cv2.imread(images_left[t]), (image_size[1], image_size[0])), 
                   cv2.resize(cv2.imread(images_right[t]), (image_size[1], image_size[0]))]
            images = torch.from_numpy(np.stack(images, 0)).permute(0,3,1,2)

            # read corresponding depths
            depths = [np.load(depth_left[t]), np.load(depth_right[t])]
            depths = torch.from_numpy(np.stack(depths, 0))

            intrinsics = .8 * torch.as_tensor(intrinsics_vec)
            self.data.append((t, images, depths, intrinsics))

    def collate_fn(self, data):
        t_batch = [d[0] for d in data]
        imgs_batch = [d[1] for d in data]
        dps_batch = [d[2] for d in data]
        intrinsics_batch = [d[3] for d in data]
        return t_batch, imgs_batch, dps_batch, intrinsics_batch

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]