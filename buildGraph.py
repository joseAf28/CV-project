import numpy as np
import scipy
import algorithmsPart2 as alg
import FrameGraphPart2 as fg
import perspectivePart2 as pt



cams_info_data = scipy.io.loadmat('office/cams_info_no_extr.mat')
kp_data = scipy.io.loadmat('office/kp.mat')
wrld_data = scipy.io.loadmat('office/wrld_info.mat')



### Convert the data to the appropriate format
num_images = 10

rgb_shape = cams_info_data['cams_info'][0][0]['rgb'][0][0].shape
rgb_type = cams_info_data['cams_info'][0][0]['rgb'][0][0].dtype

depth_shape = cams_info_data['cams_info'][0][0]['depth'][0][0].shape
depth_type = cams_info_data['cams_info'][0][0]['depth'][0][0].dtype

conf_shape = cams_info_data['cams_info'][0][0]['conf'][0][0].shape
conf_type = cams_info_data['cams_info'][0][0]['conf'][0][0].dtype


rgb_images = np.zeros((rgb_shape[0], rgb_shape[1], rgb_shape[2], num_images), dtype=rgb_type)
depth_images = np.zeros((depth_shape[0], depth_shape[1], num_images), dtype=depth_type)
conf_images = np.zeros((conf_shape[0], conf_shape[1], num_images), dtype=conf_type)

fx_vec = np.zeros((num_images), dtype=np.float64)
fy_vec = np.zeros((num_images), dtype=np.float64)

for i in range(num_images):
    rgb_images[..., i] = cams_info_data['cams_info'][i][0]['rgb'][0][0]
    depth_images[..., i] = cams_info_data['cams_info'][i][0]['depth'][0][0]
    conf_images[..., i] = cams_info_data['cams_info'][i][0]['conf'][0][0]
    
    fx_vec[i] = cams_info_data['cams_info'][i][0]['focal_lenght'][0,0][0,0]
    fy_vec[i] = cams_info_data['cams_info'][i][0]['focal_lenght'][0,0][0,0]


cx = rgb_shape[1] / 2
cy = rgb_shape[0] / 2


### Preprocess depth
def preprocess_depth(depth, conf, threshold=0.5):
    
    mask = conf > threshold
    depth_filtered = np.where(mask, depth, 0)
    return depth_filtered


depth_images_filtered = np.zeros_like(depth_images)
intrinsecs = []

for i in range(num_images):
    depth_images_filtered[..., i] = preprocess_depth(depth_images[..., i], conf_images[..., i])
    intrinsecs.append((fx_vec[i], fy_vec[i], cx, cy))


### convert the depth maps to point clouds


def depth_to_point_cloud(depth, rgb, intrinsics):
    
    fx, fy, cx, cy = intrinsics
    
    y_indices, x_indices = np.indices(depth.shape)
    
    x = (x_indices - cx) * depth / fx
    y = (y_indices - cy) * depth / fy
    z = depth
    
    points = np.stack([x, y, z], axis=-1)

    rgb = rgb.reshape(-1, 3)
    points = points.reshape(-1, 3)
    points_cloud = np.concatenate([points, rgb], axis=-1)
    
    return points_cloud


points_clouds = []

for i in range(num_images):
    cloud = depth_to_point_cloud(depth_images_filtered[..., i], rgb_images[..., i], intrinsecs[i])
    points_clouds.append(cloud)




### create the Graph to extract the keypoint descriptor and features

nodes = fg.initialize_graph(kp_data, depth_images_filtered, rgb_images, intrinsecs)


# for node in nodes:
#     print(node)
#     print(node.stats)
#     print()


node_a = nodes[0]
node_b = nodes[1]

matches = alg.matching_optional(node_a.desc, node_b.desc, 0.9)

print(node_a.desc.shape)
print(node_a.depth_pts.shape)
print(len(matches))

print(matches)
print(matches.shape)

### use the matched keypoints to find the corresponding 3D points

def get_corresponding_points(kp1, kp2, matches, depth1, depth2, intrinsics1, intrinsics2):
    fx1, fy1, cx1, cy1 = intrinsics1
    fx2, fy2, cx2, cy2 = intrinsics2
    
    kp1_matches = kp1[matches[:, 0]]
    kp2_matches = kp2[matches[:, 1]]
    
    u1, v1 = kp1_matches[:, 0], kp1_matches[:, 1]
    u2, v2 = kp2_matches[:, 0], kp2_matches[:, 1]
    
    z1 = depth1[v1.astype(int), u1.astype(int)]
    z2 = depth2[v2.astype(int), u2.astype(int)]
    
    ## depth points should be greater than 0 (== 0 means depth information with no confidence)
    valid_mask = (z1 > 0) & (z2 > 0)
    
    u1, v1, z1 = u1[valid_mask], v1[valid_mask], z1[valid_mask]
    u2, v2, z2 = u2[valid_mask], v2[valid_mask], z2[valid_mask]
    
    x1 = (u1 - cx1) * z1 / fx1
    y1 = (v1 - cy1) * z1 / fy1
    x2 = (u2 - cx2) * z2 / fx2
    y2 = (v2 - cy2) * z2 / fy2
    
    points1 = np.stack([x1, y1, z1], axis=-1)
    points2 = np.stack([x2, y2, z2], axis=-1)
    
    return points1, points2



points_a, points_b = get_corresponding_points(node_a.kp, node_b.kp, matches,node_a.depth_pts, node_b.depth_pts, \
                                                node_a.intrinsics, node_b.intrinsics)

print(points_a.shape)
print(points_b.shape)

def estimate_transformation_pc(proj1, proj2):
    # Compute centroids
    centroid_1 = np.mean(proj1, axis=0)
    centroid_2 = np.mean(proj2, axis=0)
    # Center the points
    proj1_centered = proj1 - centroid_1
    proj2_centered = proj2 - centroid_2
    # Compute covariance matrix
    H = proj1_centered.T @ proj2_centered
    # SVD
    U, S, Vt = np.linalg.svd(H)
    R_mat = Vt.T @ U.T
    # Handle reflection
    if np.linalg.det(R_mat) < 0:
        Vt[-1, :] *= -1
        R_mat = Vt.T @ U.T
    # Compute translation
    t = centroid_2 - R_mat @ centroid_1
    return R_mat, t


def estimate_affine_transformation_svd(proj1, proj2):
    """
    Estimate affine transformation (rotation, translation, shearing)
    that maps proj1 to proj2 using SVD.

    Parameters:
    - proj1: numpy array of shape (N, 3)
    - proj2: numpy array of shape (N, 3)

    Returns:
    - A: 3x3 affine transformation matrix
    - t: translation vector of size (3,)
    """
    N = proj1.shape[0]
    if proj2.shape[0] != N:
        raise ValueError("Point sets must have the same number of points.")
    
    # Step 1: Compute centroids
    centroid_1 = np.mean(proj1, axis=0)
    centroid_2 = np.mean(proj2, axis=0)
    
    # Step 2: Center the points
    proj1_centered = proj1 - centroid_1
    proj2_centered = proj2 - centroid_2
    
    # Step 3: Construct matrix M and vector b
    M = np.zeros((3 * N, 12))
    b = np.zeros((3 * N,))
    
    for i in range(N):
        p = proj1_centered[i]
        q = proj2_centered[i]
        M[3 * i] = [p[0], p[1], p[2], 1, 0, 0, 0, 0, 0, 0, 0, 0]
        M[3 * i + 1] = [0, 0, 0, 0, p[0], p[1], p[2], 1, 0, 0, 0, 0]
        M[3 * i + 2] = [0, 0, 0, 0, 0, 0, 0, 0, p[0], p[1], p[2], 1]
        b[3 * i] = q[0]
        b[3 * i + 1] = q[1]
        b[3 * i + 2] = q[2]
    
    # Step 4: Perform SVD on M
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    
    # Step 5: Compute the least-squares solution
    # The least-squares solution is given by V * S_inv * U^T * b
    S_inv = np.diag(1 / S)
    x = Vt.T @ S_inv @ U.T @ b
    
    # Step 6: Extract A and t
    A = x[:9].reshape(3, 3)
    t = x[9:]
    
    return A, t



A, t = estimate_affine_transformation_svd(points_a, points_b)

print(A)
print(t)


#### Now I have the same shit that in Part1.2
### but with this transformation MSAC