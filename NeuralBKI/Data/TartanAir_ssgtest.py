import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

# based on the neighborhood dataset
class TartanAirDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_dir = os.path.join(root_dir, 'image_left')
        self.depth_dir = os.path.join(root_dir, 'depth_left')
        # !!!getting pose directly from pose_left.txt BUT we need pose output from 
        # DROID-SLAM
        self.pose_file = os.path.join(root_dir, 'pose_left.txt') 

        # !!!getting timestamps directly from timestamps.txt BUT we need timestamps output from 
        # DROID-SLAM
        self.timestamps_file = os.path.join(root_dir, 'timestamps.txt')
        
        self.timestamps = []
        self.poses = []
        with open(self.timestamps_file, 'r') as f:
            for line in f:
                self.timestamps.append(line.strip())
        with open(self.pose_file, 'r') as f:
            for line in f:
                pose = [float(x) for x in line.strip().split()]
                self.poses.append(pose)

    def __len__(self):
        return len(self.timestamps)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.timestamps[idx] + '.png')
        depth_path = os.path.join(self.depth_dir, self.timestamps[idx] + '.pfm')
        img = Image.open(img_path).convert('RGB')
        depth = np.array(Image.open(depth_path))
        pose = self.poses[idx]
        if self.transform:
            img = self.transform(img)
        return {'image': img, 'depth': depth, 'pose': pose}
