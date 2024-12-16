import numpy as np
import scipy
import algorithms as alg
import FrameGraph as fg
import perspective as pt
import pickle
import tqdm

import os
import glob



### load the graph
with open("ISRwall/input_1/graph1.pkl", "rb") as f:
    nodes = pickle.load(f)


### print the nodes
# for node in nodes:
#     print(node)
#     print(node.connections.keys())
#     print(node.stats.keys())
#     print()



## homographies from the reference frame to all the other frames
composite_homographies = fg.compute_composite_homographies_dfs(nodes, reference_index=0)



### obtain the paths of the images
folder_path = "ISRwall/input_1/images"
image_extension = ['*.jpg']

image_files = []
for ext in image_extension:
    image_files.extend(glob.glob(os.path.join(folder_path, ext)))

image_files = sorted(image_files)

initial_image_path = image_files[0]
images_path = image_files[1:]


initial_image = pt.image_to_matrix(initial_image_path)
width, height = (20000,10000)
start=2000

H_cumulative = np.eye(3)

dst = np.full((height, width, initial_image.shape[2] if initial_image.ndim == 3 else 1), 0, dtype=initial_image.dtype)
dst = pt.warp_perspective_full(initial_image, H_cumulative, dst)

pt.matrix_to_image(dst, f"images/final_mosaic0.jpg")



for i, image_path in enumerate(tqdm.tqdm(images_path)):
    
    i = i + 1
    
    H_cumulative = composite_homographies[i]
    
    img2 = pt.image_to_matrix(image_path)
    dst = pt.warp_perspective_full(img2, H_cumulative, dst)
    pt.matrix_to_image(dst, f"images/final_mosaic{i}.jpg")



####? Things to do:

### 1. Refine the homographies at end of the process
### 2. Better way to find the best path: sum of the mean error from the RANSAC, is not that helpful
### 3. Other criteria to find the best path: number of inliers, variance of the error, ...
### 4. Group all the choices in a separate file