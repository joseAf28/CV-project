from data_loader import load_data
from point_cloud import compute_point_cloud, plot_point_cloud, save_point_cloud

def main():
    # Load data
    keypoints, cams_info, reference_frame = load_data()

    # Extract depth map, focal length, and RGB image
    depth_map = reference_frame['depth'][0, 0]
    focal_length = reference_frame['focal_lenght'][0, 0][0]
    rgb_image = reference_frame['rgb'][0, 0]  # RGB image

    # Compute principal point offsets
    cx = rgb_image.shape[1] / 2
    cy = rgb_image.shape[0] / 2

    # Compute point cloud with RGB colors
    point_cloud, colors = compute_point_cloud(depth_map, focal_length, cx, cy, rgb_image)

    # Visualize the point cloud with colors
    # plot_point_cloud(point_cloud, colors=colors, title="Colored 3D Point Cloud")

    # Save the point cloud with colors to a PLY file to open and observe in MeshLab
    save_point_cloud(point_cloud, "output_point_cloud.ply", colors=colors)

if __name__ == "__main__":
    main()
