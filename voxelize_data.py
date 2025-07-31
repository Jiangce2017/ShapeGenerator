import trimesh
import os
import os.path as osp
import matplotlib.pyplot as plt
import shutil
import numpy as np
from utils import voxelize_stl, normalized_mesh_size

dataset_root_dir = '/scratch/jc14407/datasets'
root_dir = osp.join(dataset_root_dir, 'abc_0000_stl2_v00')
root_des = osp.join(dataset_root_dir, 'abc_voxelized_20')
face_count_threshold = 50000  # Set your threshold for face count

if os.path.exists(root_des):
    shutil.rmtree(root_des)
os.makedirs(root_des, exist_ok=True)
grid_size = 20

triangle_counts = []
count = 1
countSelected = 0

for stl_folder in os.listdir(root_dir):
    files = os.path.join(root_dir, stl_folder)
    for f in os.listdir(files):
        if not f.endswith('.stl'):
            continue
        stl_path = os.path.join(files, f)
        mesh = trimesh.load_mesh(stl_path)
        triangle_count = len(mesh.faces)
        triangle_counts.append(triangle_count)
        
        if triangle_count < face_count_threshold:
            countSelected += 1
            vox = voxelize_stl(stl_path,grid_size=grid_size)
            file = f.split('.')[0]
            stl_des = os.path.join(root_des, f"{file}_grid{grid_size}.npy")
            np.save(stl_des, vox)
            print(f"{countSelected}: voxelized with {triangle_count} faces")

        else:
            print(f"{count}: skipped {f} with {triangle_count} faces")
        count += 1
            

    # stl_path = os.path.join(root_dir, file)

    # mesh = trimesh.load_mesh(stl_path)
    # vox = voxelize_stl(stl_path,grid_size=grid_size)
    # file = file.split('.')[0]
    # stl_des = os.path.join(root_des, file)
    # np.save(stl_des, vox)
    # print(f"{count}: voxelized with {len(mesh.faces)} faces")
    # count += 1

# plt.figure(figsize=(10, 6))
# plt.hist(triangle_counts, bins=50, color='skyblue', edgecolor='black')
# plt.title('Distribution of Triangle Counts in STL Files')
# plt.xlabel('Number of Triangles')
# plt.ylabel('Number of Models')
# plt.grid(True)
# plt.tight_layout()
# plt.show()
       
