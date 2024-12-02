import numpy as np
import os
import scipy.io


### YOLO MANIPULATION STUFF ###

def load_yolo(directory_path):
    ## return a dictionary with:
    ## key - id
    ## value - coordinates, nb_frame

    elements = os.listdir(directory_path)
    elements = sorted(elements)

    len_elements = len(elements)


    tensor_yolo_id = []
    tensor_yolo_xyxy = []

    for i, element in enumerate(elements):
        data_yolo = scipy.io.loadmat(f"{directory_path}/{element}")
        
        data_yolo_id = data_yolo['id']
        data_yolo_xyxy = data_yolo['xyxy']
        
        tensor_yolo_xyxy.append(data_yolo_xyxy)
        tensor_yolo_id.append(data_yolo_id)


    np.unique(tensor_yolo_id[0])

    unique_ids = np.unique(tensor_yolo_id[0])

    for i in range(1, len(tensor_yolo_id)):
        unique_ids = np.union1d(unique_ids, np.unique(tensor_yolo_id[i]))

    ### determine the coordinates for each id

    track_coordinates = {id: [] for id in unique_ids}

    for i in range(len(tensor_yolo_id)):
        for j in range(len(tensor_yolo_id[i])):
            id = tensor_yolo_id[i][j][0]
            coordinates = tensor_yolo_xyxy[i][j]
            
            x_mean = (coordinates[0] + coordinates[2]) / 2.0
            y_mean = (coordinates[1] + coordinates[3]) / 2.0
            
            coords_mean = np.array([x_mean, y_mean, i])
            
            track_coordinates[id].append(coords_mean)


    for key in track_coordinates.keys():
        track_coordinates[key] = np.array(track_coordinates[key])

    return track_coordinates



def coordinates2image(track_coordinates, card):
    
    width, height = card

    track_coordinates_int = track_coordinates.astype(int)

    ## background white, track black
    # image = 255*np.ones((height, width, 3), dtype=np.uint8)
    # for i in range(len(track_coordinates_int)):
    #     x = track_coordinates_int[i][0]
    #     y = track_coordinates_int[i][1]
    #     image[y, x] = 0
        
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(len(track_coordinates_int)):
        x = track_coordinates_int[i][0]
        y = track_coordinates_int[i][1]
        image[y, x] = 255

    return image




### LINEAR ALGEBRA STUFF ###
### FROM mint-lab repository (known from the slides provided by the professor) ###

def getPerspectiveTransform(src, dst):
    if len(src) == len(dst):
        # Make homogeneous coordiates if necessary
        if src.shape[1] == 2:
            src = np.hstack((src, np.ones((len(src), 1), dtype=src.dtype)))
        if dst.shape[1] == 2:
            dst = np.hstack((dst, np.ones((len(dst), 1), dtype=dst.dtype)))

        # Solve 'Ax = 0'
        A = []
        for p, q in zip(src, dst):
            A.append([0, 0, 0, q[2]*p[0], q[2]*p[1], q[2]*p[2], -q[1]*p[0], -q[1]*p[1], -q[1]*p[2]])
            A.append([q[2]*p[0], q[2]*p[1], q[2]*p[2], 0, 0, 0, -q[0]*p[0], -q[0]*p[1], -q[0]*p[2]])
        _, _, Vt = np.linalg.svd(A, full_matrices=True)
        x = Vt[-1]

        # Reorganize `x` as a matrix
        H = x.reshape(3, -1) / x[-1] # Normalize the last element as 1
        return H



def warpPerspective2(src, H, dst_size):
    # Generate an empty image
    width, height = dst_size
    channel = src.shape[2] if src.ndim > 2 else 1
    dst = np.zeros((height, width, channel), dtype=src.dtype)

    # Copy a pixel from `src` to `dst` (backward mapping)
    H_inv = np.linalg.inv(H)
    for qy in range(height):
        for qx in range(width):
            p = H_inv @ [qx, qy, 1]
            px, py = int(p[0]/p[-1] + 0.5), int(p[1]/p[-1] + 0.5)
            if px >= 0 and py >= 0 and px < src.shape[1] and py < src.shape[0]:
                dst[qy, qx] = src[py, px]
    return dst