import algorithms as alg
import perspective as pt
import FrameGraph as fg
import numpy as np
import scipy
import glob
import pickle
import os



### obtain the paths to the keypoints
# folder_path = "ISRwall/input_1/keypoints"


folder_path = "volley/input"


keypoints_extension = ['*.mat']

keypoints_files = []
for ext in keypoints_extension:
    keypoints_files.extend(glob.glob(os.path.join(folder_path, ext)))

keypoints_files = sorted(keypoints_files)
# keypoints_files = keypoints_files[:300]

for i in range(len(keypoints_files)):
        
    kp, desc = fg.load_keypoints(keypoints_files[i])
    
    if kp is None or desc is None:
        keypoints_files[i] = None

keypoints_files = [x for x in keypoints_files if x is not None]

# keypoints_files = keypoints_files[:450]


### path to the reference image and the reference index

reference_image = "volley/reference/img_ref.jpg"
reference_file = "volley/reference/kp_ref.mat"



##! initialize the graph: create the nodes
nodes = fg.initialize_graph(reference_file, keypoints_files)
reference_index = 0

##! compute the edges
nodes = fg.compute_edges(nodes, reference_index)

for node in nodes:
    print(node)
    # print(node.connections)
    print(node.stats)
    print()



##! save the graph

# with open("ISRwall/input_1/graph1.pkl", "wb") as f:
#     pickle.dump(nodes, f)


with open("volley/graph1.pkl", "wb") as f:
    pickle.dump(nodes, f)

