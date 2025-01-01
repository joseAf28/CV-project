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

# def get_corresponding_points(kp1, kp2, matches, depth1, depth2, intrinsics1, intrinsics2):
    # fx1, fy1, cx1, cy1 = intrinsics1
    # fx2, fy2, cx2, cy2 = intrinsics2
    
    # kp1_matches = kp1[matches[:, 0]]
    # kp2_matches = kp2[matches[:, 1]]
    
    # u1, v1 = kp1_matches[:, 0], kp1_matches[:, 1]
    # u2, v2 = kp2_matches[:, 0], kp2_matches[:, 1]
    
    # z1 = depth1[v1.astype(int), u1.astype(int)]
    # z2 = depth2[v2.astype(int), u2.astype(int)]
    
    # ## depth points should be greater than 0 (== 0 means depth information with no confidence)
    # valid_mask = (z1 > 0) & (z2 > 0)
    
    # u1, v1, z1 = u1[valid_mask], v1[valid_mask], z1[valid_mask]
    # u2, v2, z2 = u2[valid_mask], v2[valid_mask], z2[valid_mask]
    
    # x1 = (u1 - cx1) * z1 / fx1
    # y1 = (v1 - cy1) * z1 / fy1
    # x2 = (u2 - cx2) * z2 / fx2
    # y2 = (v2 - cy2) * z2 / fy2
    
    # points1 = np.stack([x1, y1, z1], axis=-1)
    # points2 = np.stack([x2, y2, z2], axis=-1)
    
    # return points1, points2



# points_a, points_b = get_corresponding_points(node_a.kp, node_b.kp, matches,node_a.depth_pts, node_b.depth_pts, \
#                                                 node_a.intrinsics, node_b.intrinsics)



points3D_a = node_a.points_3D
points3D_b = node_b.points_3D

print(points3D_a.shape)
print(points3D_b.shape)
print(points3D_a)

best_inliers = alg.MSAC(matches, node_a.points_3D, node_b.points_3D, {'MSAC_max_iter': 1000, 'MSAC_threshold': 5.0, 'MSAC_confidence': 0.99})
# best_inliers = alg.RANSAC(matches, node_a.points_3D, node_b.points_3D, {'RANSAC_max_iter': 1000, 'RANSAC_inlier_threshold': 7.0})


print(len(best_inliers))

#### Now I have the same shit that in Part1.2
### but with this transformation MSAC


# PARAMS = {
#     'MSAC_max_iter': 1000, 
#     'MSAC_threshold': 5.0, 
#     'MSAC_confidence': 0.99
    
    
# }

PARAMS = {
    'match_threshold': 0.9,
    'edges_num_neighbors': 3,
    'edges_inliers_threshold': 18, # 12 also works well
    'node_reference_index': 0,
    'RANSAC_inlier_threshold': 2.0,
    'RANSAC_max_iter': 700,
    'MSAC_max_iter': 1000,
    'MSAC_threshold': 5.0,
    'MSAC_confidence': 0.99,
}

nodes = fg.compute_edges(nodes, )