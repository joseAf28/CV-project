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


reference_index = 0

## homographies from the reference frame to all the other frames
composite_homographies, path_lenghts, path_costs, graph  = fg.compute_composite_homographies(nodes, reference_index=reference_index)




# ### Plot of the path lenghts
# plt.figure()
# plt.plot(path_lenghts, '.')
# plt.xlabel("Frame index")
# plt.ylabel("Path length")
# plt.grid()
# plt.savefig("volley/path_lenghts.png")


# ### Plot of the path lenghts
# plt.figure()
# plt.plot(path_costs, '.')
# plt.xlabel("Frame index")
# plt.ylabel("Path Cost")
# plt.grid()
# plt.savefig("volley/path_costs.png")  


# ### Plot the graph
# plt.figure()
# fg.plot_graph(graph)


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

## split mat_files into kp and yolo
kp_files = []
yolo_files = []

for mat_file in mat_files:
    if 'kp' in mat_file:
        kp_files.append(mat_file)
    else:
        yolo_files.append(mat_file)


print(kp_files)
print(len(kp_files))
print(yolo_files)
print(len(yolo_files))



for i in range(len(image_files)):
    
    kp, desc = fg.load_keypoints(mat_files[i])
    
    if kp is None or desc is None:
        iamge_files[i] = None


image_files = [x for x in image_files if x is not None]

### files of the reference image
initial_image_path = "volley/reference/img_ref.jpg"

images_path = image_files



# print(mat_files)


# ### Yolo data

# data_yolo = scipy.io.loadmat("volley/volley.mat")
# dir_yolo = data_yolo.keys()

# frames_yolo_vec = [key for key in dir_yolo if 'frame' in key]



# # images_path = images_path[:450]

# initial_image = pt.image_to_matrix(initial_image_path)
# width, height = (2000,1000)



# H_cumulative = np.eye(3, dtype=np.float64)

# dst = np.full((height, width, initial_image.shape[2] if initial_image.ndim == 3 else 1), 0, dtype=initial_image.dtype)
# dst = pt.warp_perspective_full(initial_image, H_cumulative, dst)

# pt.matrix_to_image(dst, f"volley/images/final_mosaic0.jpg")


# num_images = len(images_path)
# print(len(images_path))


# H_array = np.zeros(len(images_path), dtype=object)

# for i, image_path in enumerate(tqdm.tqdm(images_path)):
    
#     i = i + 1
    
#     if i == num_images:
#         break
    
#     H_cumulative = composite_homographies[i]
    
#     img2 = pt.image_to_matrix(image_path)
#     dst = pt.warp_perspective_full(img2, H_cumulative, dst)
#     pt.matrix_to_image(dst, f"volley/images/final_mosaic{i}.jpg")
    
    
#     ## Output Data
#     H_array[i] = (H_cumulative, i)
    
#     try:
#         data_xyxy = data_yolo[frames_yolo_vec[i]]['xyxy']
#         data_id = data_yolo[frames_yolo_vec[i]]['id']
#     except:
#         continue
    
    
#     data_xyxy = data_xyxy[0][0]
#     data_id = data_id[0][0]
    
#     cm_x = (data_xyxy[:,0] + data_xyxy[:,2]) / 2
#     cm_y = (data_xyxy[:,1] + data_xyxy[:,3]) / 2
    
#     cm_points = np.column_stack((cm_x, cm_y))
#     cm_points_hom = np.column_stack((cm_points, np.ones((cm_points.shape[0], 1)))).T
    
#     transformed_cm_points_hom = (H_cumulative @ cm_points_hom).T
#     transformed_cm_points_hom /= transformed_cm_points_hom[:,2][:, None]
    
#     width = data_xyxy[:,2] - data_xyxy[:,0]
#     height = data_xyxy[:,3] - data_xyxy[:,1]
    
#     transformed_cm_points = transformed_cm_points_hom[:, :2]
    
#     bounding_boxes_transformed = np.column_stack((transformed_cm_points[:,0] - width/2, transformed_cm_points[:,1] - height/2, \
#                                         transformed_cm_points[:,0] + width/2, transformed_cm_points[:,1] + height/2))
    
    
#     struct_yolo = {"xyxy": bounding_boxes_transformed, "id": data_id}
    
    
#     number = str(i).zfill(4)
#     scipy.io.savemat(f"volley/output/yolooutput_{number}.mat", struct_yolo)


# struct_H = {"H": H_array}
# scipy.io.savemat("volley/output/homographies.mat", struct_H)


