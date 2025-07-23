import trimesh
import os
import matplotlib.pyplot as plt
import shutil

root_dir = "./datasets/abc_cleaned"
root_des = "./datasets/abc_low"

triangle_counts = []
count = 1
countUnder50000 = 0

for file in os.listdir(root_des):
    stl_path = os.path.join(root_des, file)
    mesh = trimesh.load_mesh(stl_path)
    triangle_counts.append(len(mesh.faces))
    print(count)
    count += 1

plt.figure(figsize=(10, 6))
plt.hist(triangle_counts, bins=50, color='skyblue', edgecolor='black')
plt.title('Distribution of Triangle Counts in STL Files')
plt.xlabel('Number of Triangles')
plt.ylabel('Number of Models')
plt.grid(True)
plt.tight_layout()
plt.show()
       
