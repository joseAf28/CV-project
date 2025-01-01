import numpy as np
import scipy 


## getting homography algorithm


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


# def estimate_affine_transformation_svd(proj1, proj2):
def estimate_affine_transformation_svd(src, dst):
    ### Estimate affine transformation (rotation, translation, shearing)
    ### that maps src to dst using SVD.

    N = src.shape[0]
    if dst.shape[0] != N:
        raise ValueError("Point sets must have the same number of points.")
    
    centroid_1 = np.mean(src, axis=0)
    centroid_2 = np.mean(dst, axis=0)
    
    proj1_centered = src - centroid_1
    proj2_centered = dst - centroid_2
    
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
    
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    
    S_inv = np.diag(1 / S)
    x = Vt.T @ S_inv @ U.T @ b
    
    A = x[:9].reshape(3, 3)
    t = x[9:]
    
    return A, t



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

