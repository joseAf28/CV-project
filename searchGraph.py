import numpy as np
import scipy
import algorithms as alg
import FrameGraph as fg
import perspective as pt
import pickle
import tqdm
import matplotlib.pyplot as plt

import os
import glob



### load the graph
# with open("ISRwall/input_1/graph1.pkl", "rb") as f:
#     nodes = pickle.load(f)
    
with open("volley/graph1.pkl", "rb") as f:
    nodes = pickle.load(f)


### print the nodes
# for node in nodes:
#     print(node)
#     print(node.connections.keys())
#     print(node.stats.keys())
#     print()

reference_index = 163
## homographies from the reference frame to all the other frames
composite_homographies, path_lenghts  = fg.compute_composite_homographies_2('', nodes, reference_index=reference_index)


### Plot of the path lenghts
plt.figure()
plt.plot(path_lenghts, '.')
plt.xlabel("Frame index")
plt.ylabel("Path length")
plt.grid()
plt.savefig("volley/path_lenghts.png")

### obtain the paths of the images
# folder_path = "ISRwall/input_1/images"
folder_path = "volley/input"


image_extension = ['*.jpg']
mat_extenstion = ['*.mat']

image_files = []
mat_files = []

for ext in image_extension:
    image_files.extend(glob.glob(os.path.join(folder_path, ext)))


for ext in mat_extenstion:
    mat_files.extend(glob.glob(os.path.join(folder_path, ext)))

image_files = sorted(image_files)
mat_files = sorted(mat_files)


for i in range(len(image_files)):
    
    kp, desc = fg.load_keypoints(mat_files[i])
    
    if kp is None or desc is None:
        iamge_files[i] = None


image_files = [x for x in image_files if x is not None]


initial_image_path = image_files[reference_index]
images_path = image_files[0:reference_index] + image_files[reference_index+1:]

images_path = images_path[:450]


initial_image = pt.image_to_matrix(initial_image_path)
width, height = (2000,1000)



H_cumulative = np.eye(3)

dst = np.full((height, width, initial_image.shape[2] if initial_image.ndim == 3 else 1), 0, dtype=initial_image.dtype)
dst = pt.warp_perspective_full(initial_image, H_cumulative, dst)

pt.matrix_to_image(dst, f"volley/images/final_mosaic0.jpg")

# pt.matrix_to_image(initial_image, f"images/final_mosaic0.jpg")

num_images = len(images_path)
print(len(images_path))


for i, image_path in enumerate(tqdm.tqdm(images_path)):
    
    i = i + 1
    
    if i == num_images:
        break
    
    H_cumulative = composite_homographies[i]
    
    img2 = pt.image_to_matrix(image_path)
    dst = pt.warp_perspective_full(img2, H_cumulative, dst)
    pt.matrix_to_image(dst, f"volley/images/final_mosaic{i}.jpg")



####? Things to do:

### 1. Refine the homographies at end of the process
### 2. Better way to find the best path: sum of the mean error from the RANSAC, is not that helpful
### 3. Other criteria to find the best path: number of inliers, variance of the error, ...
### 4. Group all the choices in a separate file