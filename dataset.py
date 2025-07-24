import os
from torch.utils.data import Dataset
from utils import voxelize_stl
import torch
import trimesh
import numpy as np

class ABCDataset(Dataset):
    def __init__(self, root_dir, size=10000, voxelized = False):
        """
        Args:
            root_dir (str): Folder containing .stl files
        """
        self.root_dir = root_dir
        self.voxelized = voxelized
        if voxelized:
            self.modelPaths = [
                os.path.join(root_dir, f)
                for f in os.listdir(root_dir)
                if f.endswith('.npy')
            ]
        else:
            self.modelPaths = [
                os.path.join(root_dir, f)
                for f in os.listdir(root_dir)
                if f.endswith('.stl')
            ][:size]

    def __len__(self):
        return len(self.modelPaths)

    def __getitem__(self, idx):
        path = self.modelPaths[idx]
        if self.voxelized:
            vox = np.load(self.modelPaths[idx])
            vox_tensor = torch.tensor(vox, dtype=torch.float32).unsqueeze(-1)
            print("loaded voxel")
        else:
            mesh = trimesh.load_mesh(path)
            vox = voxelize_stl(path)  # shape: [D, H, W] numpy array
            vox_tensor = torch.tensor(vox, dtype=torch.float32).unsqueeze(-1)  # → [D, H, W, 1]
            print(f"Item voxelized with {len(mesh.faces)} faces")
        return {'model': vox_tensor}