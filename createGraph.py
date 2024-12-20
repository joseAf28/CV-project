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


keypoints_files = keypoints_files[:450]

# ### vector of filenames
# # filenames_vec = [f"ISRwall/input_1/keypoints/kp_000{i}.mat" for i in range(1, 10)]

##! initialize the graph: create the nodes
nodes = fg.initialize_graph(keypoints_files)

##! compute the edges
# nodes = fg.compute_delaunay_edges(nodes)
# nodes = fg.compute_edges(nodes)
nodes = fg.compute_proximity_edges(nodes)


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

