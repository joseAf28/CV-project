import numpy as np
import os
import scipy.io
import imageio.v2 as imageio

# Converts an image to a NumPy matrix representation
def image_to_matrix(image_path):
    img_matrix = imageio.imread(image_path) # Load the image into a NumPy array
    # If the image has an alpha channel (RGBA), convert it to RGB
    if img_matrix.shape[-1] == 4:  # RGBA image
        img_matrix = img_matrix[:, :, :3]  # Remove the alpha channel
    return img_matrix

# Loads YOLO data from a directory and organizes it by object ID
def load_yolo(directory_path="."):
    # Returns a dictionary with:
    # Key - object ID
    # Value - coordinates and corresponding frame number

    # List all files in the directory and sort them
    elements = os.listdir(directory_path)
    elements = sorted(elements)

    # Initialize tensors for YOLO data
    tensor_yolo_id = []
    tensor_yolo_xyxy = []

    # Process each YOLO detection file
    for i, element in enumerate(elements):
        data_yolo = scipy.io.loadmat(f"{directory_path}/{element}")
        
        # Extract IDs and bounding box coordinates
        data_yolo_id = data_yolo['id']
        data_yolo_xyxy = data_yolo['xyxy']
        
        tensor_yolo_xyxy.append(data_yolo_xyxy)
        tensor_yolo_id.append(data_yolo_id)

    # Identify all unique object IDs
    unique_ids = np.unique(tensor_yolo_id[0])
    for i in range(1, len(tensor_yolo_id)):
        unique_ids = np.union1d(unique_ids, np.unique(tensor_yolo_id[i]))

    # Initialize a dictionary to store coordinates for each ID
    track_coordinates = {id: [] for id in unique_ids}
    # Collect coordinates for each object ID across frames
    for i in range(len(tensor_yolo_id)):
        for j in range(len(tensor_yolo_id[i])):
            id = tensor_yolo_id[i][j][0]
            coordinates = tensor_yolo_xyxy[i][j]
            
            # Compute the mean (center) of the bounding box
            x_mean = (coordinates[0] + coordinates[2]) / 2.0
            y_mean = (coordinates[1] + coordinates[3]) / 2.0
            
            coords_mean = np.array([x_mean, y_mean, i])
            
            track_coordinates[id].append(coords_mean)

    # Convert lists to NumPy arrays
    for key in track_coordinates.keys():
        track_coordinates[key] = np.array(track_coordinates[key])

    return track_coordinates

# Converts object coordinates to an image representation
def coordinates2image(track_coordinates, card, color, large=False):
    width, height = card
    track_coordinates_int = track_coordinates.astype(int)
    
    # Initialize the image canvas
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Validate that coordinates are within the image bounds
    valid_mask = (
        (track_coordinates_int[:, 0] >= 0) & (track_coordinates_int[:, 0] < width) &
        (track_coordinates_int[:, 1] >= 0) & (track_coordinates_int[:, 1] < height)
    )
    track_coordinates_valid = track_coordinates_int[valid_mask]

    # Paint valid pixels with the specified color
    image[track_coordinates_valid[:, 1], track_coordinates_valid[:, 0]] = color

    # Optionally expand to neighboring pixels (manual dilation)
    if large:
        dilated_image = np.zeros_like(image)
        for dx in range(-large, large + 1):
            for dy in range(-large, large + 1):
                shifted_y = np.clip(track_coordinates_valid[:, 1] + dy, 0, height - 1)
                shifted_x = np.clip(track_coordinates_valid[:, 0] + dx, 0, width - 1)
                dilated_image[shifted_y, shifted_x] = color
        image = dilated_image

    return image

### FROM mint-lab repository (known from the slides provided by the professor) ###
# Computes the perspective transformation matrix
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

# Warps an image using a perspective transformation matrix
def warpPerspective2(src, H, dst_size, color=[255, 255, 255]):
    width, height = dst_size
    H_inv = np.linalg.inv(H)

    # Generate a grid of coordinates in the destination space
    qx, qy = np.meshgrid(np.arange(width), np.arange(height))
    q_coords = np.stack([qx.ravel(), qy.ravel(), np.ones_like(qx).ravel()])

    # Transform destination coordinates back to the source space
    p_coords = H_inv @ q_coords
    p_coords /= p_coords[2]   # Normalize homogeneous coordinates
    px, py = p_coords[0].astype(int), p_coords[1].astype(int)

    # Mask for valid coordinates
    valid_mask = (px >= 0) & (py >= 0) & (px < src.shape[1]) & (py < src.shape[0])

    # Create the destination image
    dst = np.zeros((height, width, src.shape[2]), dtype=src.dtype)
    dst[qy.ravel()[valid_mask], qx.ravel()[valid_mask]] = src[py[valid_mask], px[valid_mask]]

    return dst