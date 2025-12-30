from PIL import Image
from tqdm import tqdm
import numpy as np
import os

import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 64
img_transforms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
])

def get_torch_xyza(lidar_depth, azimuth, zenith):
    """
    Convert LiDAR depth map to XYZA format using azimuth and zenith angles.
    Applies masking to zero out background points and normalizes valid points.
    """
    if lidar_depth.dim() == 3:
        lidar_depth = lidar_depth.squeeze(0)
        
    x = lidar_depth * torch.sin(-azimuth[:, None]) * torch.cos(-zenith[None, :])
    y = lidar_depth * torch.cos(-azimuth[:, None]) * torch.cos(-zenith[None, :])
    z = lidar_depth * torch.sin(-zenith[None, :])
    
    # Create mask for valid depth (less than 50.0)
    mask = (lidar_depth < 50.0).float()
    
    # Apply mask to zero out background and normalize valid points
    x = (x * mask) / 50.0
    y = (y * mask) / 50.0
    z = (z * mask) / 50.0

    a = mask
    
    xyza = torch.stack((x, y, z, a))
    return xyza

class CubesAndShperesDataset(Dataset):
    """
    Custom Dataset for loading RGB images and LiDAR depth maps of cubes and spheres.
    """
    def __init__(self, root_dir):
        self.classes = ["cubes", "spheres"]
        self.root_dir = root_dir
        self.rgb = []
        self.lidar = []
        self.class_idxs = []

        for class_idx, class_name in enumerate(self.classes):
            print(f"Constructing Dataset for class {class_name}")
            class_path = os.path.join(self.root_dir, class_name)
            rgb_path = os.path.join(class_path, "rgb")
            lidar_path = os.path.join(class_path, "lidar")
            
            # Load azimuth and zenith
            azimuth_path = os.path.join(class_path, "azimuth.npy")
            zenith_path = os.path.join(class_path, "zenith.npy")
            
            if not os.path.exists(azimuth_path) or not os.path.exists(zenith_path):
                print(f"Warning: Azimuth/Zenith not found in {class_path}")
                continue
                
            azimuth = torch.from_numpy(np.load(azimuth_path)).to(device)
            zenith = torch.from_numpy(np.load(zenith_path)).to(device)
            
            if not os.path.exists(rgb_path) or not os.path.exists(lidar_path):
                print(f"Warning: Directory not found: {rgb_path} or {lidar_path}")
                continue

            files = sorted([f for f in os.listdir(lidar_path) if f.endswith('.npy')])
            
            for file_name in tqdm(files, desc="Files for class"):
                file_id = os.path.splitext(file_name)[0]
                rgb_file = os.path.join(rgb_path, file_id + ".png")
                lidar_file = os.path.join(lidar_path, file_name)
                
                if os.path.exists(rgb_file):
                    try:
                        rbg_img = Image.open(rgb_file)
                        rbg_img = img_transforms(rbg_img).to(device)
                        
                        lidar_depth = np.load(lidar_file)
                        lidar_depth = torch.from_numpy(lidar_depth).to(torch.float32).to(device)
                        lidar_xyza = get_torch_xyza(lidar_depth, azimuth, zenith)
                        
                        self.rgb.append(rbg_img)
                        self.lidar.append(lidar_xyza)
                        self.class_idxs.append(torch.tensor(class_idx, dtype=torch.float32)[None].to(device))
                    except Exception as e:
                        print(f"Error loading {file_id}: {e}")

    def __len__(self):
        return len(self.class_idxs)

    def __getitem__(self, idx):
        rbg_img = self.rgb[idx]
        lidar_depth = self.lidar[idx]
        class_idx = self.class_idxs[idx]
        return rbg_img, lidar_depth, class_idx