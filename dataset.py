import os
from torch.utils.data import Dataset

class ABCDataset(Dataset):
    def __init__(self, root_dir, size = 10000):
        """
        Args:
            root_dir (str): Root folder of ABC dataset (e.g., 'abc/')
        """
        self.root_dir = root_dir
        self.stl_paths = []
        self.ids = []
        count = 0

        # Traverse each subfolder
        for folder in os.listdir(root_dir):
            if (count >= size): break
            folder_path = os.path.join(root_dir, folder)
            if os.path.isdir(folder_path):
                stl_files = [f for f in os.listdir(folder_path) if f.endswith('.stl')]
                if len(stl_files) == 1:
                    stl_path = os.path.join(folder_path, stl_files[0])
                    self.stl_paths.append(stl_path)
                    self.ids.append(folder)
                elif len(stl_files) > 1:
                    raise ValueError(f"Multiple STL files in {folder_path}")
                # else: silently skip folders without .stl
            count += 1

    def __len__(self):
        return len(self.stl_paths)

    def __getitem__(self, idx):
        return {
            'id': self.ids[idx],
            'stl_path': self.stl_paths[idx]
        }
