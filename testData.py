import trimesh
import os
import matplotlib.pyplot as plt
import shutil
import numpy as np
from utils import voxelize_stl

root_dir = "./datasets/abc_low"
root_des = "./datasets/abc_voxelized"

triangle_counts = []
count = 1
countUnder50000 = 0

for file in os.listdir(root_dir):
    stl_path = os.path.join(root_dir, file)
    mesh = trimesh.load_mesh(stl_path)
    vox = voxelize_stl(stl_path)
    file = file.split('.')[0]
    stl_des = os.path.join(root_des, file)
    np.save(stl_des, vox)
    print(f"{count}: voxelized with {len(mesh.faces)} faces")
    count += 1

# plt.figure(figsize=(10, 6))
# plt.hist(triangle_counts, bins=50, color='skyblue', edgecolor='black')
# plt.title('Distribution of Triangle Counts in STL Files')
# plt.xlabel('Number of Triangles')
# plt.ylabel('Number of Models')
# plt.grid(True)
# plt.tight_layout()
# plt.show()
       
