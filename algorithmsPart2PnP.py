import numpy as np
import scipy 
from scipy.spatial import KDTree


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


import cv2

### matching algorithms

def matching_optional(desc1, desc2, threshold=0.9):
    
    ## calculate the distances between each query descriptor and each train descriptor: i -> query, j -> train
    distances = scipy.spatial.distance.cdist(desc1, desc2, 'euclidean') 
    sorted_indices = np.argsort(distances, axis=1)[:, :2]
    
    first_min_indices = sorted_indices[:, 0]
    second_min_indices = sorted_indices[:, 1]

    first_min_distances = distances[np.arange(distances.shape[0]), first_min_indices]
    second_min_distances = distances[np.arange(distances.shape[0]), second_min_indices]

    condition = first_min_distances < threshold * second_min_distances
    matches = np.column_stack((np.where(condition)[0], first_min_indices[condition]))
    
    return matches



def ratio_test_matching(source_descriptors, target_descriptors, ratio=0.75):

    tree = KDTree(target_descriptors)
    distances, indices = tree.query(source_descriptors, k=2)
    
    good_matches = []
    for i in range(len(source_descriptors)):
        if distances[i][0] < ratio * distances[i][1]:
            good_matches.append((i, indices[i][0]))
    return good_matches


def mutual_nearest_neighbor_matching(source_descriptors, target_descriptors):

    tree_target = KDTree(target_descriptors)
    distances_st, indices_st = tree_target.query(source_descriptors, k=1)
    
    tree_source = KDTree(source_descriptors)
    distances_ts, indices_ts = tree_source.query(target_descriptors, k=1)
    
    mutual_matches = []
    for src_idx, tgt_idx in enumerate(indices_st):
        if indices_ts[tgt_idx] == src_idx:
            mutual_matches.append((src_idx, tgt_idx))
    
    return mutual_matches


def hybrid_matching(source_descriptors, target_descriptors, ratio=0.9):
    ###Combines Lowe's Ratio Test with Mutual Nearest Neighbors.
    
    ratio_matches = ratio_test_matching(source_descriptors, target_descriptors, ratio=ratio)
    
    ratio_match_set = set(ratio_matches)
    
    mutual_matches = set(mutual_nearest_neighbor_matching(source_descriptors, target_descriptors))
    
    hybrid_matches = list(ratio_match_set.intersection(mutual_matches))
    
    return np.array(hybrid_matches)



def RANSAC(matches, points3D_1, points3D_2, PARAMS):
    
    inlier_threshold = PARAMS['RANSAC_inlier_threshold']
    max_iter = PARAMS['RANSAC_max_iter']
    
    best_inliers = []
    
    counter = 0
    
    if matches.shape[0] < 4:
        return best_inliers, None
    
    while counter < max_iter:
        
        sampled_matches = matches[np.random.choice(matches.shape[0], 4, replace=False)]
        
        points1 = points3D_1[sampled_matches[:, 0]]
        points2 = points3D_2[sampled_matches[:, 1]]
        
        A, t = estimate_affine_transformation_svd(points1, points2)
        
        transformed_pts = np.dot(A, points3D_1[matches[:,0]].T).T + t
        
        residuals = np.linalg.norm(transformed_pts - points3D_2[matches[:,1]], axis=1)
        inliers = np.where(residuals < inlier_threshold)[0]
        
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
        
        counter += 1
        
    return best_inliers



def RANSAC_PNP(object_points, image_points, intrinsics):
    
    threshold = 5.0
    confidence = 0.99
    
    intrinsics_mat = np.array([[intrinsics[0], 0, intrinsics[2]], [0, intrinsics[1], intrinsics[3]], [0, 0, 1]])
    
    retval, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, image_points, intrinsics_mat, None, confidence, threshold)
    
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    
    if retval:
        return inliers, rotation_matrix, tvec
    else:
        return None, None, None









def MSAC(matches, points3D_1, points3D_2, PARAMS):

    max_iterations = PARAMS['MSAC_max_iter']
    threshold = PARAMS['MSAC_threshold']
    confidence = PARAMS['MSAC_confidence']

    num_points = matches.shape[0]
    best_H = None
    best_inliers = []
    best_score = float('inf')  # MSAC minimizes the total cost
    iterations = 0
    best_confidence = 0.0

    if matches.shape[0] < 4:
        return best_inliers, None


    while iterations < max_iterations and best_confidence < confidence:
        
        sampled_matches = matches[np.random.choice(matches.shape[0], 4, replace=False)]
        
        points1 = points3D_1[sampled_matches[:, 0]]
        points2 = points3D_2[sampled_matches[:, 1]]
        
        A, t = estimate_affine_transformation_svd(points1, points2)
        
        transformed_pts = np.dot(A, points3D_1[matches[:,0]].T).T + t
        
        residuals = np.linalg.norm(transformed_pts - points3D_2[matches[:,1]], axis=1)
        
        # new cost function: tries to minimize the residuals
        costs = np.where(residuals < threshold, residuals**2, threshold**2)
        total_cost = np.sum(costs)
        
        
        if total_cost < best_score:
            best_score = total_cost
            
            inliers = np.where(residuals < threshold)[0]
            best_inliers = matches[inliers]
            
            # Update confidence
            inlier_ratio = len(inliers) / num_points
            if inlier_ratio > 0:
                best_confidence = 1 - (1 - inlier_ratio ** 4) ** iterations if iterations > 0 else 0
            else:
                best_confidence = 0
        
        iterations += 1

    return best_inliers



##### Implement ICP algorithm

def apply_transformation(points, transformation):

    num_points = points.shape[0]
    homogeneous_points = np.hstack((points, np.ones((num_points, 1))))
    
    transformed_homogeneous = (transformation @ homogeneous_points.T).T
    transformed_points = transformed_homogeneous[:, :3]
    
    return transformed_points


def find_closest_points(source, target):
    tree = KDTree(target)
    distances, indices = tree.query(source, k=1)
    return distances, indices



def iterative_closest_point(source_points, target_points, initial_transformation=np.eye(4),
                            max_iterations=50, tolerance=1e-5):
    ### Performs Iterative Closest Point (ICP) to refine the transformation.

    transformation = initial_transformation.copy()
    
    transformed_source = apply_transformation(source_points, transformation)
    
    prev_error = float('inf')
    for i in range(max_iterations):
        
        distances, indices = find_closest_points(transformed_source, target_points)
    
        corresponding_target = target_points[indices]
        
        R, t = estimate_affine_transformation_svd(transformed_source, corresponding_target)
        
        delta_transformation = np.eye(4)
        delta_transformation[:3, :3] = R
        delta_transformation[:3, 3] = t
        
        transformation = delta_transformation @ transformation
        
        transformed_source = apply_transformation(source_points, transformation)
        
        mean_error = np.mean(distances)
        # print(f"Iteration {i+1}: Mean Error = {mean_error}")
        
        if abs(prev_error - mean_error) < tolerance:
            # print("Convergence reached.")
            break
        prev_error = mean_error
    
    A = transformation[:3, :3]
    t = transformation[:3, 3]
    return A, t