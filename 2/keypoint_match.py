import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


def match_keypoints_ransac(desc1, desc2, kp1, kp2, threshold=0.75, max_iter=1000, inlier_threshold=2.0):
    """
    Match keypoints between two sets using RANSAC for robustness.

    :param desc1: Descriptors from frame 1 (NxD).
    :param desc2: Descriptors from frame 2 (MxD).
    :param kp1: Keypoints from frame 1 (Nx2).
    :param kp2: Keypoints from frame 2 (Mx2).
    :param threshold: Ratio test threshold for Lowe's ratio test.
    :param max_iter: Maximum number of RANSAC iterations.
    :param inlier_threshold: Distance threshold to consider a match as an inlier.
    :return: Inliers (list of matches) and the best homography matrix.
    """
    distances = cdist(desc1, desc2)  # Pairwise distances
    matches = []
    for i, row in enumerate(distances):
        sorted_indices = np.argsort(row)
        if row[sorted_indices[0]] < threshold * row[sorted_indices[1]]:
            matches.append((i, sorted_indices[0]))
    matches = np.array(matches)

    # RANSAC
    best_inliers = []
    best_H = None

    for _ in range(max_iter):
        # Randomly sample 4 matches
        sampled_matches = matches[np.random.choice(matches.shape[0], 4, replace=False)]
        src_pts = kp1[sampled_matches[:, 0]]
        dst_pts = kp2[sampled_matches[:, 1]]

        # Compute homography matrix
        H, status = compute_homography(src_pts, dst_pts)

        # Count inliers
        inliers = []
        for match in matches:
            pt1 = kp1[match[0]]
            pt2 = kp2[match[1]]

            pt1_hom = np.append(pt1, 1)  # Convert to homogeneous coordinates
            transformed_pt = H @ pt1_hom
            transformed_pt /= transformed_pt[2]  # Normalize

            dist = np.linalg.norm(transformed_pt[:2] - pt2)
            if dist < inlier_threshold:
                inliers.append(match)

        # Update best model if more inliers are found
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_H = H

    return best_inliers, best_H


def compute_homography(src_pts, dst_pts):
    """
    Compute a homography matrix using point correspondences.

    :param src_pts: Source points (Nx2).
    :param dst_pts: Destination points (Nx2).
    :return: Homography matrix (3x3).
    """
    assert len(src_pts) >= 4, "At least 4 points are required to compute a homography."
    A = []
    for (x, y), (xp, yp) in zip(src_pts, dst_pts):
        A.append([-x, -y, -1, 0, 0, 0, x * xp, y * xp, xp])
        A.append([0, 0, 0, -x, -y, -1, x * yp, y * yp, yp])
    A = np.array(A)

    # Solve A.h = 0 using SVD
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    return H


def plot_matches(img1, img2, kp1, kp2, matches, title="Keypoint Matches"):
    """
    Plot matches between two images.

    :param img1: First image (numpy array).
    :param img2: Second image (numpy array).
    :param kp1: Keypoints from frame 1 (Nx2).
    :param kp2: Keypoints from frame 2 (Mx2).
    :param matches: Matched keypoints (Px2).
    :param title: Title of the plot.
    """
    plt.figure(figsize=(15, 8))
    combined_img = np.hstack((img1, img2))
    plt.imshow(combined_img, cmap='gray')

    for match in matches:
        pt1 = kp1[match[0]]
        pt2 = kp2[match[1]] + [img1.shape[1], 0]  # Shift X for second image
        plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'r-', alpha=0.6)

    plt.title(title)
    plt.axis('off')
    plt.show()
    