import numpy as np
import scipy
import algorithms as alg
import FrameGraph as fg
import perspective as pt
import pickle
import tqdm



### load the graph

with open("ISRwall/input_1/graph.pkl", "rb") as f:
    nodes = pickle.load(f)


for node in nodes:
    print(node)
    print(node.connections.keys())
    print(node.stats.keys())
#     print()



### homographies from the reference frame to all the other frames
composite_homographies = fg.compute_composite_homographies(nodes, reference_index=0)


## compute the mosaic

initial_image_path = 'ISRwall/input_1/images/img_0001.jpg'
initial_image = pt.image_to_matrix(initial_image_path)
width, height = (20000,10000)
start=2000
dst = np.full((height, width, initial_image.shape[2] if initial_image.ndim == 3 else 1), 0, dtype=initial_image.dtype)

for y in range(initial_image.shape[0]):
    for x in range(initial_image.shape[1]):
        if 0 <= x+12000 < width and 0 <= y+6000 < height:
            dst[y+6000, x+12000] = initial_image[y, x]

pt.matrix_to_image(dst, f"final_mosaic.jpg")



## check first the homographies
H_cumulative = np.eye(3)

for i in tqdm.tqdm(range(1, 8)):
    
    # matches = nodes[i-1].inliers[i]
    
    # kp1 = nodes[i-1].keypoints[matches[:, 0]]
    # kp2 = nodes[i].keypoints[matches[:, 1]]
    
    # H = alg.getPerspectiveTransform(kp2, kp1)
    # H_cumulative = np.dot(H_cumulative, H)

    
    H_cumulative = composite_homographies[i]
    
    img2 = pt.image_to_matrix(f"ISRwall/input_1/images/img_000{i+1}.jpg")
    
    dst = pt.warp_perspective_full(img2, H_cumulative, dst)

    
    pt.matrix_to_image(dst, f"final_mosaic{i}.jpg")






# ### Now add the refinement step to the homographies



