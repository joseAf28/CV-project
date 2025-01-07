import algorithms as alg
import perspective as pt
import FrameGraph as fg
import numpy as np
import scipy
import glob
import pickle
import os
import argparse
import tqdm


PARAMS = {
    'match_threshold': 0.25,
    'edges_num_neighbors': 3,
    'edges_inliers_threshold': 18, # 12 also works well
    'node_reference_index': 0,
    
    'RANSAC_inlier_threshold': 2.0,
    'RANSAC_max_iter': 700,
    
    'MSAC_max_iter': 1000,
    'MSAC_threshold': 4.0,
    'MSAC_confidence': 0.99,
}


if __name__ == "__main__":
    
    ###! Read the input arguments
    parser = argparse.ArgumentParser(description="Use structure: ref_dir input_dir1 output_dir1 input_dir2 output_dir2 ...")

    parser.add_argument('ref_dir', type=str, help='Reference directory')
    parser.add_argument('dirs', nargs='+', help='Pairs of input and output directories')

    parser.add_argument('--verbose', action='store_true', help='Enable verbose mode')

    args = parser.parse_args()

    ref_dir = args.ref_dir
    dirs = args.dirs
    verbose = args.verbose

    if len(dirs) % 2 != 0:
        raise ValueError("The number of input and output directories must be even.")

    input_output_pairs = [(dirs[i], dirs[i + 1]) for i in range(0, len(dirs), 2)]

    if verbose:
        print(f"Reference directory: {ref_dir}")
        for i, (input_dir, output_dir) in enumerate(input_output_pairs):
            print(f"Pair {i + 1}: Input directory: {input_dir}, Output directory: {output_dir}")
    
    
    
    ###! Create the graph
    
    input_folder_paths = [x[0] for x in input_output_pairs]
    output_folder_paths = [x[1] for x in input_output_pairs]
    
    # for i in range(len(input_output_pairs)):
        
    #     print("input path: ", input_folder_paths[i], "output path: ", output_folder_paths[i])
        
    # for i, folder_path in enumerate(input_folder_paths):
    #     print(f"Input folder {i}: {folder_path}", "Output folder: ", output_folder_paths[i])
    
    
    
    input_output_path_2 = {input_folder_paths[i]: output_folder_paths[i] for i in range(len(input_folder_paths))}
    # print("Input Output Path 2: ", input_output_path_2)
    
    input_output_path = dict()
    
    mat_extension = ['*.mat']
    img_extension = ['*.jpg']
    
    mat_files = []
    img_files = []
    for i, folder_path in enumerate(input_folder_paths):
        for ext in mat_extension:
            file = glob.glob(os.path.join(folder_path, ext))
            mat_files.extend(file)
            
            for file in mat_files:
                if 'yolo' in file:
                    # print("Yolo file: ", file, "Output folder: ", output_folder_paths[i], "input folder: ", folder_path)
                    input_output_path[file] = output_folder_paths[i]
    
        for ext in img_extension:
            file = glob.glob(os.path.join(folder_path, ext))
            img_files.extend(file)
            
    
    mat_files = sorted(mat_files)
    img_files = sorted(img_files)
    
    kp_files = []
    yolo_files = []
    
    for mat_file in mat_files:
        if 'kp' in mat_file:
            kp_files.append(mat_file)
        else:
            yolo_files.append(mat_file)
    
    
    for i in range(len(kp_files)):
        kp, desc = fg.load_keypoints(kp_files[i])
            
        if kp is None or desc is None:
            kp_files[i] = None


    kp_files = [x for x in kp_files if x is not None]
    
    
    
    print("yolo files: ", yolo_files)
    
    # print("Kp_files init:", kp_files)
    
    reference_image = os.path.join(ref_dir, "img_ref.jpg")
    reference_file = os.path.join(ref_dir, "kp_ref.mat")
    
    # ##! initialize the graph: create the nodes
    # nodes = fg.initialize_graph(reference_file, kp_files)
    
    # ##! compute the edges
    # nodes = fg.compute_edges(nodes, PARAMS)
    
    
    # ###! Save the graph
    # graph_dir = os.path.join(ref_dir, "..")
    # graph_file = os.path.join(graph_dir, "graph.pkl")
    
    # with open(graph_file, "wb") as f:
    #     pickle.dump(nodes, f)
    
    
    with open("volley2/graph.pkl", "rb") as f:
        nodes = pickle.load(f)
    
    ###! Compute the composite homographies
    composite_homographies, path_lenghts, path_costs, graph  = fg.compute_composite_homographies(nodes, PARAMS)
    
    
    H_cumulative = np.eye(3, dtype=np.float64)
    
    # print("Input Output Path: ", input_output_path)
    
    # ##? For debugging the final mosaic
    # width, height = (2000,1000)
    # initial_image = pt.image_to_matrix(reference_image)
    # dst = np.full((height, width, initial_image.shape[2] if initial_image.ndim == 3 else 1), 0, dtype=initial_image.dtype)
    # dst = pt.warp_perspective_full(initial_image, H_cumulative, dst)
    # pt.matrix_to_image(dst, f"volley2/images/final_mosaic0.jpg")


    num_images = len(img_files)

    # H_dict = {output_folder_paths[i] : [] for i in range(len(output_folder_paths))}
    
    H_dict = {img_files[i] : [] for i in range(len(img_files))}

    for i, img_file in enumerate(tqdm.tqdm(img_files, desc="Processing images")):
        
        i = i + 1
        
        if i == num_images+1:
            break
        
        H_cumulative = composite_homographies[i]
        
        
        # ##? For debugging the final mosaic
        # img2 = pt.image_to_matrix(img_file)
        # dst = pt.warp_perspective_full(img2, H_cumulative, dst)
        # pt.matrix_to_image(dst, f"volley2/images/final_mosaic{i}.jpg")
        
        
        ## Output Data
        
        data_xyxy, data_id = fg.load_yolo(yolo_files[i-1])
        
        if data_xyxy is None or data_id is None or data_xyxy.shape[1] != 4:
            continue
        
        
        cm_x = (data_xyxy[:,0] + data_xyxy[:,2]) / 2
        cm_y = (data_xyxy[:,1] + data_xyxy[:,3]) / 2
        
        cm_points = np.column_stack((cm_x, cm_y))
        cm_points_hom = np.column_stack((cm_points, np.ones((cm_points.shape[0], 1)))).T
        
        transformed_cm_points_hom = (H_cumulative @ cm_points_hom).T
        transformed_cm_points_hom /= transformed_cm_points_hom[:,2][:, None]
        
        width = data_xyxy[:,2] - data_xyxy[:,0]
        height = data_xyxy[:,3] - data_xyxy[:,1]
        
        transformed_cm_points = transformed_cm_points_hom[:, :2]
        
        bounding_boxes_transformed = np.column_stack((transformed_cm_points[:,0] - width/2, transformed_cm_points[:,1] - height/2, \
                                            transformed_cm_points[:,0] + width/2, transformed_cm_points[:,1] + height/2))
        
        
        struct_yolo = {"xyxy": bounding_boxes_transformed, "id": data_id}
        
        
        filename = f"yolooutput_{str(i).zfill(4)}"
        # output_dir = input_output_path[yolo_files[i-1]]
        
        
        yolo_file_path = yolo_files[i-1]
        parts = yolo_file_path.split('/')
        parts[1] = parts[1].replace('input', 'output')
        
        output_dir = '/'.join(parts[:2])
        print("Output dir: ", output_dir)
        
        # print("output_dir: ", output_dir, "filename: ", filename)
        
        scipy.io.savemat(os.path.join(output_dir, f"{filename}.mat"), struct_yolo)
        
        # outputs
        H_dict[img_file].append((H_cumulative, i))
        

    for input_dir, H_array in H_dict.items():
        print("Input dir: ", input_dir)
        
        parts = input_dir.split('/')
        parts[1] = parts[1].replace('input', 'output')
        input_dir_mod = ('/'.join(parts[:2])) + "/homographies.mat"
        
        struct_H = {"H": H_array}
        scipy.io.savemat(input_dir_mod, struct_H)
        
        # struct_H = {"H": H_array}
        # scipy.io.savemat(os.path.join(output_dir, "homographies.mat"), struct_H)
