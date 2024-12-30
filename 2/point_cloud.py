import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def compute_point_cloud(depth_map, focal_length, cx, cy, rgb_image=None):
    """
    Compute a 3D point cloud from a depth map, with optional RGB colors.

    :param depth_map: 2D array of depth values (H x W).
    :param focal_length: Scalar focal length (assume fx = fy).
    :param cx: Principal point x-coordinate (usually image width / 2).
    :param cy: Principal point y-coordinate (usually image height / 2).
    :param rgb_image: 3D array (H x W x 3) of RGB image (optional).
    :return: Nx3 array of 3D points, Nx3 array of RGB colors (if provided).
    """
    height, width = depth_map.shape
    points_3d = []
    colors = [] if rgb_image is not None else None

    for v in range(height):
        for u in range(width):
            Z = depth_map[v, u]
            if Z > 0:  # Ignore invalid depth values
                X = (u - cx) * Z / focal_length[0]
                Y = (v - cy) * Z / focal_length[0]
                points_3d.append([X, Y, Z])
                
                if rgb_image is not None:
                    colors.append(rgb_image[v, u])  # Get RGB color

    if colors is not None:
        return np.array(points_3d), np.array(colors)
    return np.array(points_3d)


def plot_point_cloud(points_3d, colors=None, title="3D Point Cloud"):
    """
    Plot a 3D point cloud with optional color.

    :param points_3d: Nx3 array of 3D points.
    :param colors: Nx3 array of RGB colors (optional, normalized between 0 and 1).
    :param title: Title of the plot.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Use colors if provided, otherwise default to blue
    if colors is not None:
        if np.max(colors) > 1.0:  # Normalize if not already
            colors = colors / 255.0
        ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c=colors, s=1)
    else:
        ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='b', s=1)
    
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


def save_point_cloud(points_3d, filename, colors=None):
    """
    Save a 3D point cloud to a PLY file, with optional RGB colors.

    :param points_3d: Nx3 array of 3D points.
    :param filename: Path to save the PLY file.
    :param colors: Nx3 array of RGB colors (optional, normalized between 0 and 1 or in 0–255 range).
    """
    with open(filename, 'w') as file:
        # Write the header
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {len(points_3d)}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        if colors is not None:
            file.write("property uchar red\n")
            file.write("property uchar green\n")
            file.write("property uchar blue\n")
        file.write("end_header\n")

        # Write the points and optionally the colors
        for i, point in enumerate(points_3d):
            if colors is not None:
                r, g, b = colors[i]
                # Ensure colors are in the 0–255 range
                if np.max(colors) <= 1.0:
                    r, g, b = int(r * 255), int(g * 255), int(b * 255)
                file.write(f"{point[0]} {point[1]} {point[2]} {r} {g} {b}\n")
            else:
                file.write(f"{point[0]} {point[1]} {point[2]}\n")

    print(f"Point cloud saved to {filename}")

