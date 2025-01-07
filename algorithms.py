import numpy as np
import scipy 


print_flag = False

## getting homography algorithm

def getPerspectiveTransform(src, dst):
    if len(src) == len(dst):
        # Make homogeneous coordinates if necessary
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

        H = x.reshape(3, -1) / x[-1] # Normalize the last element as 1
        return H




### matching algorithms

def matching_optional(desc1, desc2, threshold=0.25):
    
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



def RANSAC(matches, kp1, kp2, PARAMS):
    
    inlier_threshold = PARAMS['RANSAC_inlier_threshold']
    max_iter = PARAMS['RANSAC_max_iter']
    
    best_inliers = []
    
    counter = 0
    
    if matches.shape[0] < 4:
        return best_inliers, None
    
    while counter < max_iter:
        
        sampled_matches = matches[np.random.choice(matches.shape[0], 4, replace=False)]
        
        H = getPerspectiveTransform(kp1[sampled_matches[:, 0]], kp2[sampled_matches[:, 1]])

        src_pts_hom = np.column_stack((kp1[matches[:, 0]], np.ones((matches.shape[0], 1)))).T
        transformed_pts_hom = (H @ src_pts_hom).T
        transformed_pts_hom /= transformed_pts_hom[:, 2][:, None]

        dst_pts_all = kp2[matches[:, 1]]
        dists = np.linalg.norm(transformed_pts_hom[:, :2] - dst_pts_all, axis=1)

        inliers = matches[dists < inlier_threshold]
        
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            
            
        if print_flag:
            print("Number of iteractions: ", number_iteractions," epsilon: ",  epsilon,  " best inliers: ", len(best_inliers) , " inliers nb: ", len(inliers))
        
        
        counter += 1
        
    return best_inliers


def MSAC(matches, kp1, kp2, PARAMS):

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
        # Step 1: Randomly sample 4 correspondences
        sampled_matches = matches[np.random.choice(matches.shape[0], 4, replace=False)]
        
        # homography estimation
        H = getPerspectiveTransform(kp1[sampled_matches[:, 0]], kp2[sampled_matches[:, 1]])

        # transform the source points
        src_pts_hom = np.column_stack((kp1[matches[:, 0]], np.ones((matches.shape[0], 1)))).T
        transformed_pts_hom = (H @ src_pts_hom).T
        transformed_pts_hom /= transformed_pts_hom[:, 2][:, None]

        dst_pts_all = kp2[matches[:, 1]]
        
        # residuals
        residuals = np.linalg.norm(transformed_pts_hom[:, :2] - dst_pts_all, axis=1)
        
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

