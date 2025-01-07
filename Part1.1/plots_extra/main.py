import src
#import cv2 as cv
import numpy as np
import scipy 
import tqdm
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os

if __name__ == "__main__":

    # Flags to control which parts of the code to execute
    #teste_1 = False  # Convert one camera image to Google Maps perspective
    combined_trajectories = False  # Combine trajectories across frames
    yolo_in_gm = True # Save transformed YOLO outputs
    all_frames_in_gm = True  # Generate video frames in Google Maps perspective
    save_gift = False 
    large = 2  #!!! -> False if you dont want dilate  |  Control dilation of trajectory points 

    # Folder containing input data
    folder_name = "CTownAirport_1.1"

    # Load keypoint matches between the video and Google Maps image
    #data_maps = scipy.io.loadmat(f'{folder_name}/kp_gmaps.mat')
    data_maps = scipy.io.loadmat(f'kp_gmaps.mat')
 
    data_maps_vec = data_maps['kp_gmaps']

    # Check if the output directory exists, and create it if not
    directory = ''
    #if not os.path.exists(directory):
    #    os.makedirs(directory)
    
    # Extract pixel and map coordinates
    pixel_coordinates = data_maps_vec[:, 0:2]
    maps_coordinates = data_maps_vec[:, 2:]

    # Compute the perspective transformation matrix and save it into a .mat file
    H = src.getPerspectiveTransform(pixel_coordinates, maps_coordinates)
    scipy.io.savemat('homography.mat', {'H': H})
    
    # Test transformation of one image to Google Maps perspective
    """
    if teste_1:
        matrix = src.image_to_matrix(f'{folder_name}/images/img_0001.jpg')
        img_maps = src.warpPerspective2(matrix, H, (938, 1347))
        src.matrix_to_image(img_maps,  "img_0001_in_gp.jpg")"""

    # Assumed image dimensions
    card_src = (640, 352)  # Camera frame dimensions
    card_dst = (938, 1347)  # Google Maps image dimensions
    
#-----------------------------------------------------------------------------------------------
    if combined_trajectories:

        # Load the YOLO data
        #directory_path = f"{folder_name}/yolo"
        #track_coordinates = src.load_yolo(directory_path)
        track_coordinates = src.load_yolo()

        len_track_coordinates = len(track_coordinates) # Compute the length of track_coordinates
        idx_keys = track_coordinates.keys() # Extract the keys (track IDs)
        n = len(idx_keys)  # Compute the total number of keys (total unique tracks)

        # Generate colors for trajectories
        colors = plt.cm.tab20(np.linspace(0, 1, n))  # Use 'tab20' colormap to generate 5 colors
        # Initialize combined trajectory images for both perspectives
        image_all_combined_video = np.zeros((card_src[1], card_src[0], 3), dtype=np.uint8)
        image_all_combined_map = np.zeros((card_dst[1], card_dst[0], 3), dtype=np.uint8)
        
        for i in tqdm.tqdm(range(len_track_coordinates)):
            idx = list(idx_keys)[i]
            color_one = colors[i, :-1]*255  # Extract RGB color
            track_coordinates_one = track_coordinates[idx][:, :-1]
            
            # Generate trajectory image for camera frame
            track_coordinates_one_img = src.coordinates2image(track_coordinates_one, card_src, color_one, large)
            # Warp trajectory image to Google Maps perspective
            track_coordinates_maps = src.warpPerspective2(track_coordinates_one_img, H, card_dst, color_one)

            # Combine trajectory images
            image_all_combined_map += track_coordinates_maps
            image_all_combined_video += track_coordinates_one_img

        # Overlay trajectories on the Google Maps image
        #image_map = src.image_to_matrix(f'{folder_name}/airport_CapeTown_aerial.png')  # imagem do Google Maps
        image_map = src.image_to_matrix(f'airport_CapeTown_aerial.png')  # imagem do Google Maps
        mask = np.any(image_all_combined_map > 0, axis=-1)  #onde estão as trajetórias
        image_map[mask] = image_all_combined_map[mask] 

        # Save images
        #imageio.imwrite(f'{directory}/map_with_trajectories.jpg', image_map)
        #imageio.imwrite(f'{directory}/track_coordinates_maps_all.jpg', image_all_combined_map)
        #imageio.imwrite(f'{directory}/track_coordinates_video_all.jpg', image_all_combined_video)
        imageio.imwrite(f'map_with_trajectories.jpg', image_map)
        imageio.imwrite(f'track_coordinates_maps_all.jpg', image_all_combined_map)
        imageio.imwrite(f'track_coordinates_video_all.jpg', image_all_combined_video)

#--------------------------------------------------------------------------------------------
    if all_frames_in_gm or yolo_in_gm:
        frames_with_tracks = []  # Store transformed frames
        color_one = [255, 0, 0]  # Color for trajectory points

        # Process each frame in the sequence
        for frame_number in tqdm.tqdm(range(1, 360)):  
            # Load and transform the camera image to Google Maps perspective
            #atrix = src.image_to_matrix(f'{folder_name}/images/img_{frame_number:04}.jpg')
            matrix = src.image_to_matrix(f'img_{frame_number:04}.jpg')
            frame = src.warpPerspective2(matrix, H, (938, 1347))

            # Load YOLO data for the current frame
            #data_yolo = scipy.io.loadmat(f"{directory_path}/yolo_{frame_number:04}.mat")
            data_yolo = scipy.io.loadmat(f"yolo_{frame_number:04}.mat")
            data_yolo_id = data_yolo['id'].astype(int)
            data_yolo_xyxy = data_yolo['xyxy']


            if all_frames_in_gm:
                # Create an empty image to draw detections
                image = np.zeros((352, 640, 3), dtype=np.uint8)

                # Calculate detection centers
                centers_x = ((data_yolo_xyxy[:, 0] + data_yolo_xyxy[:, 2]) / 2).astype(int)
                centers_y = ((data_yolo_xyxy[:, 1] + data_yolo_xyxy[:, 3]) / 2).astype(int)

                # Filter valid coordinates
                valid_mask = ((centers_x >= 0) & (centers_x < 640) &(centers_y >= 0) & (centers_y < 352))
                centers_x = centers_x[valid_mask]
                centers_y = centers_y[valid_mask]

                # Draw centers on the image
                image[centers_y, centers_x] = [255, 0, 0]

                # Dilate points for better visibility
                if large:
                    for dx in range(-large, large + 1):
                        for dy in range(-large, large + 1):
                            shifted_x = np.clip(centers_x + dx, 0, 639)
                            shifted_y = np.clip(centers_y + dy, 0, 351)
                            image[shifted_y, shifted_x] = [255, 0, 0]

                # Warp detections to Google Maps perspective
                points_maps = src.warpPerspective2(image, H, card_dst, color_one)
                mask = np.any(points_maps > 0, axis=-1)  
                frame[mask] = points_maps[mask] 
                
                # Save transformed frame
                imageio.imwrite(f'output_{frame_number:04}.jpg', frame)
                frames_with_tracks.append(frame)

            if yolo_in_gm:
                # Transform the bounding box coordinates
                xyxy_transformed = np.zeros_like(data_yolo_xyxy)
                for i, (x1, y1, x2, y2) in enumerate(data_yolo_xyxy):
                    # Apply homography to the four corners of each bounding box
                    corners = np.array([[x1, y1, 1], [x2, y1, 1], [x1, y2, 1], [x2, y2, 1]])
                    transformed_corners = (H @ corners.T).T
                    transformed_corners /= transformed_corners[:, 2][:, None] 

                    # Get transformed bounding box [x_blc, y_blc, x_trc, y_trc]
                    x_min, y_min = transformed_corners[:, :2].min(axis=0)
                    x_max, y_max = transformed_corners[:, :2].max(axis=0)
                    xyxy_transformed[i] = [x_min, y_min, x_max, y_max]

                # Save transformed YOLO data in the required format
                transformed_data = {
                    'id': data_yolo_id,                         # Object IDs
                    'xyxy': xyxy_transformed,                   # Transformed bounding boxes
                    'class': data_yolo['class']                 # Object classes
                }
                scipy.io.savemat(f"yolooutput_{frame_number:04}.mat", transformed_data)

        # Save transformed frames as a GIF
        if save_gift:
            imageio.mimwrite(f'{directory}/output_with_tracks.gif', frames_with_tracks, fps=10)

