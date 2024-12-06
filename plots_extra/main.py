import src
import src
import cv2 as cv
import numpy as np
import scipy 
#import imageio
import tqdm
import matplotlib.pyplot as plt
import imageio.v2 as imageio

if __name__ == "__main__":
    
    ### Compute the homography matrix
    
    ##! Later improve homography with RANSAC
    
    ## src: pixel_coordinates
    ## dst: maps_coordinates
    
    folder_name = "CTownAirport_1.1"

    data_maps = scipy.io.loadmat(f'{folder_name}/kp_gmaps.mat')
    data_maps_vec = data_maps['kp_gmaps']
    
    pixel_coordinates = data_maps_vec[:, 0:2]
    maps_coordinates = data_maps_vec[:, 2:]

    H = src.getPerspectiveTransform(pixel_coordinates, maps_coordinates)
    

    teste_1 = False
    teste_2 = False

    ### converter imagem da camera para a prespetiva do google maps (teste que o prof pediu)
    if teste_1:
        matrix = src.image_to_matrix(f'{folder_name}/images/img_0001.jpg')
        img_maps = src.warpPerspective2(matrix, H, (938, 1347))
        src.matrix_to_image(img_maps,  "img_0001_in_gp.jpg")
    
    if teste_2:
        for i in range(1, 360):
            matrix = src.image_to_matrix(f'{folder_name}/images/img_{i:04}.jpg')
            img_maps = src.warpPerspective2(matrix, H, (938, 1347))
            src.matrix_to_image(img_maps,  f'frames_in_gp/img_{i:04}_in_gp.jpg')
            print(f'{folder_name}/images/img_{i:04}.jpg')


    ### Load the YOLO data
    directory_path = f"{folder_name}/yolo"
    track_coordinates = src.load_yolo(directory_path)

    ###! Assumption
    card_src = (640, 352)
    card_dst = (938, 1347)
    
    len_track_coordinates = len(track_coordinates)
    idx_keys = track_coordinates.keys()

    n = len(idx_keys)
    colors = plt.cm.tab20(np.linspace(0, 1, n))  # Use 'tab20' colormap to generate 5 colors
    
    image_all_combined_video = np.zeros((card_src[1], card_src[0], 3), dtype=np.uint8)
    image_all_combined_map = np.zeros((card_dst[1], card_dst[0], 3), dtype=np.uint8)
    
    all_combined_map_and_video_black = False
    all_combined_map = True
    all_frames_in_gp = False
    large = True

    if all_combined_map_and_video_black or all_combined_map:
        for i in tqdm.tqdm(range(len_track_coordinates)):
            
            idx = list(idx_keys)[i]
            color_one = colors[i, :-1]*255
            
            track_coordinates_one = track_coordinates[idx][:, :-1]
            
            track_coordinates_one_img = src.coordinates2image(track_coordinates_one, card_src, color_one, large)
            track_coordinates_maps = src.warpPerspective2(track_coordinates_one_img, H, card_dst, color_one)

            #imageio.imwrite(f'images/track_coordinates_maps_{idx}.jpg', track_coordinates_maps)
            #imageio.imwrite(f'images/track_coordinates_video_{idx}.jpg', track_coordinates_one_img)
            
            image_all_combined_map += track_coordinates_maps
            image_all_combined_video += track_coordinates_one_img

        if all_combined_map_and_video_black:
            imageio.imwrite(f'images/track_coordinates_maps_all.jpg', image_all_combined_map)
            imageio.imwrite(f'images/track_coordinates_video_all.jpg', image_all_combined_video)

        if all_combined_map:     
            image_map = src.image_to_matrix(f'{folder_name}/airport_CapeTown_aerial.png')  # imagem do Google Maps
            mask = np.any(image_all_combined_map > 0, axis=-1)  #onde estão as trajetórias
            image_map[mask] = image_all_combined_map[mask] 
            src.matrix_to_image(image_map, f"images/map_with_trajectories.jpg")

    if all_frames_in_gp:
        frames_with_tracks= []
        # Iterar pelos frames
        for frame_number in tqdm.tqdm(range(1, 360)):  
            frame_path = f"frames_in_gp/img_{frame_number:04}_in_gp.jpg"
            frame = imageio.imread(frame_path)

            data_yolo = scipy.io.loadmat(f"{directory_path}/yolo_{frame_number:04}.mat")
            data_yolo_id = data_yolo['id'].astype(int)
            data_yolo_xyxy = data_yolo['xyxy']

            image = np.zeros((352, 640, 3), dtype=np.uint8)
            for i in range(len(data_yolo_id)):
                coordinates = data_yolo_xyxy[i]
                x = int((coordinates[0] + coordinates[2]) / 2.0)
                y = int((coordinates[1] + coordinates[3]) / 2.0)
                #color_one = colors[data_yolo_id[i], :-1]*255
                color_one = [255,0,0]
                image[y, x] = color_one
                    
                if large:
                    # Coordenadas dos pixels adjacentes
                    margin = 5
                for dx in range(-margin, margin + 1):  # Intervalo horizontal
                    for dy in range(-margin, margin + 1):  # Intervalo vertical
                        new_x = x + dx
                        new_y = y + dy

                        # Verifique se o pixel está dentro dos limites da imagem
                        if 0 <= new_x < image.shape[1] and 0 <= new_y < image.shape[0]:
                            image[new_y, new_x] = color_one

            #src.matrix_to_image(image, f"t1.jpg")
            points_maps = src.warpPerspective2(image, H, card_dst, color_one)
            mask = np.any(points_maps > 0, axis=-1)  
            frame[mask] = points_maps[mask] 
            src.matrix_to_image(frame, f"t/{frame_number}.jpg")

            frames_with_tracks.append(frame)

        imageio.mimwrite(
            "images/output_with_tracks.gif",
            frames_with_tracks,
            fps=10  # Ajuste a taxa de quadros conforme necessário
        )