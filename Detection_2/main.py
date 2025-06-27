#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import sys
import cv2
import torch
import numpy as np
from pytorch3d.structures import Meshes
from pytorch3d.io import load_ply
from img_edge_prediction import img_edge_prediction
from optimization_z_torch import optimization_z_torch
from full_model import full_model
# ==================================== CONFIGURATION SECTION ====================================

debug = False
save = True
show_model = False

# Path to v4r directory
sys.path.append('/home/elisabeth/MA/3D-DAT')


import v4r_dataset_toolkit as v4r

base_path = '/home/elisabeth/MA/depth-estimation-of-transparent-objects/depth-estimation-of-transparent-objects'

# Scene-specific configuration
scene_id = 'j_005'
object_name = 'MediumBottle'  # 'BigBottle' or 'MediumBottle' as of now # Todo add more

# ==================================== END OF CONFIGURATION SECTION ====================================

# Object paths for mesh files
OBJECT_PATHS = {  # Todo add more
    'BigBottle': 'objects/BigBottle/LargeRinseFluidA_Bottle_small.ply',
    'MediumBottle': 'objects/MediumBottle/MediumRinseFluidK_Bottle_small.ply'
}

# Select object number based on object name # todo nicer
if object_name == 'BigBottle':
    object_nr = 2
elif object_name == 'MediumBottle':
    object_nr = 3

# Path to configuration file
CONFIG_PATH = 'config.cfg'

# File path separator based on the environment
split_at = '/'  # different for windows / linux

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # todo can't test

'-----------------------------------------------------------------------------------'
def extract_number(filename):
    numbers = re.findall(r'\d+', filename)
    return int(numbers[-1]) if numbers else float('inf')
# =================================================== MAIN ===================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Pose refinement and verification using differentiable rendering')
    parser.add_argument('--debug', dest='debug', default=False, action='store_true')
    parser.add_argument('--dataset_cfg', type=str, default=os.path.join(base_path, CONFIG_PATH))
    args = parser.parse_args()

    # 3D-DAT scene setup
    scene_file_reader = v4r.io.SceneFileReader.create(args.dataset_cfg)

    # !!! numpy version 1.26.4, higher throws an error
    tool_cam_poses = [pose.tf for pose in scene_file_reader.get_camera_poses(scene_id)]

    intrinsics = scene_file_reader.get_camera_info_scene(scene_id).as_numpy3x3()

    rgb_paths = scene_file_reader.get_images_rgb_path(scene_id)

    flat_mask_paths = v4r.io.get_file_list(
        os.path.join(scene_file_reader.root_dir, scene_file_reader.scenes_dir, scene_id, 'masks'), '.png')

    mask_paths = [
        [mask_path for mask_path in flat_mask_paths
         if mask_path.split('.')[-2].split(split_at)[-1].split('_')[-1] == rgb_path.split('.')[-2].split(split_at)[-1]]
        for rgb_path in rgb_paths
    ]


    object_poses = scene_file_reader.get_object_poses(scene_id)
    init_poses_tool_torch = [torch.tensor(pose, dtype=torch.float32).reshape(4, 4) for obj, pose in object_poses]

    # Load meshes using PyTorch3D
    if object_name in OBJECT_PATHS:  # todo else
        file_path = os.path.join(base_path, OBJECT_PATHS[object_name])
    else:
        print('No such object')
        sys.exit()
    vertices, faces = load_ply(file_path)
    scale_factor = 0.001
    vertices = vertices * scale_factor
    scaled_mesh_pytorch3d = Meshes(verts=[vertices.to(device)], faces=[faces.to(device)])

    # Coordinate transformation for BigBottle & MediumBottle (done manually in MeshLab)
    if object_name == 'BigBottle':

        H_fix_BigBottle_torch = torch.tensor(
            [[1, 0, 0.08, -0.007],
             [0, 1, -0.03, 0.002],
             [-0.08, 0.03, 1, 0],
             [0, 0, 0, 1]],
            dtype=torch.float32,
            device=device
        )

    else:
        H_fix_BigBottle_torch = torch.tensor(
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]],
            dtype=torch.float32,
            device=device
        )
    for rgb_idx, rgb_path in enumerate(rgb_paths):

        print("___________________________________________________________________")
        print("Rgb image: " + str(rgb_idx + 1))

        img = cv2.imread(rgb_paths[rgb_idx])

        H_I_C = tool_cam_poses[rgb_idx]
        H_I_C_torch = torch.tensor(H_I_C, dtype=torch.float32).to(device)

        img_mask_paths = mask_paths[rgb_idx]

        # Get only specified object mask #
        img_mask_path = [mask_path for mask_path in img_mask_paths
                         if mask_path.split('.')[-2].split(split_at)[-1].split('_')[-5] == object_name]
        img_mask = cv2.imread(img_mask_path[0], cv2.IMREAD_GRAYSCALE)



        # Pose object in world frame
        H_C_O_all_torch = [torch.matmul(torch.linalg.inv(H_I_C_torch), pose) for pose in init_poses_tool_torch]
        H_C_O_all_mod_torch = [
            torch.matmul(torch.linalg.inv(H_I_C_torch), torch.matmul(H_fix_BigBottle_torch, pose))
            for pose in init_poses_tool_torch
        ]

        # Use only specific object
        H_C_O_torch = H_C_O_all_torch[object_nr]
        H_C_O_mod_torch = H_C_O_all_mod_torch[object_nr]

        # Get edge image
        print('Predict edges')
        edge_image = img_edge_prediction(img, img_mask, rgb_idx, object_name, H_I_C, H_C_O_mod_torch.detach(),
                                         intrinsics, kernel_width=11, kernel_width2=11, img_path=rgb_path,
                                         img_mask_path=img_mask_path)  # todo variables


        print('Predict edges done')
        # continue  # todo remove

        # if edge image has no edges, skip entire image
        if np.max(edge_image) == 0:
            print('No points in predicted liquid mask')
            if save:
                print("saved")
                png_img = (cv2.cvtColor(img, cv2.COLOR_BGR2BGRA))
                cv2.imwrite(os.path.join("output_img", f"{object_name}{rgb_idx + 1}.png"), png_img)

                # no mask to overlay
                cv2.imwrite(os.path.join("output_img", f"{object_name}overlay{rgb_idx + 1}.png"), png_img)

                # save text file with optimal z_val
                # Define the text file path
                text_file_path = os.path.join("output_img", f"{object_name}.txt")

                # Convert line format
                new_line = f"{rgb_idx + 1}: --- \n"

                if os.path.exists(text_file_path):
                    # If file exists, read lines
                    with open(text_file_path, "r") as f:
                        lines = f.readlines()
                else:
                    # If file doesn't exist, initialize empty list
                    lines = []

                    # Ensure the list has enough lines
                while len(lines) <= rgb_idx:
                    lines.append("\n")  # Add empty lines if necessary

                    # Modify the correct line
                lines[rgb_idx] = new_line

                # Write back the modified content
                with open(text_file_path, "w") as f:
                    f.writelines(lines)
            continue

        # Convert to torch tensors and move to device # todo check if moving to device is correct, can't do here
        intrinsics_torch = torch.tensor(intrinsics, dtype=torch.float32).to(device)
        edge_image_torch = torch.tensor(edge_image, dtype=torch.float32).to(device)
        img_mask_torch = torch.tensor(img_mask, requires_grad=True, dtype=torch.float32).to(device)
        z_val_tensor = torch.tensor([0.0], requires_grad=True, device=device)  # Initial z value

        print('Call optimization z')

        # Define loss function for optimization
        def loss_function_torch(z_val):
            loss_torch, _ = optimization_z_torch(z_val, scaled_mesh_pytorch3d, intrinsics_torch, edge_image_torch,
                                                 H_C_O_mod_torch, H_I_C_torch, H_C_O_torch, img_mask_torch)
            return loss_torch


        # Adam optimizer
        optimizer = torch.optim.Adam([z_val_tensor], lr=0.001)

        # Parameters
        max_iterations = 300
        loss_history = []
        patience = 10
        loss_change_threshold = 0.001

        # Optimization loop
        for i in range(max_iterations):
            optimizer.zero_grad()
            loss = loss_function_torch(z_val_tensor)
            loss.backward()
            optimizer.step()
            print(f"Step {i + 1}, Loss: {loss.item()}, z_val: {z_val_tensor.item()}")

            # Store the current loss
            loss_history.append(loss.item())

            if loss < 0.001:
                print(f"Loss smaller than 0.001. Stopping at iteration {i + 1}")
                break

            # Keep only the last 'patience' number of losses
            if len(loss_history) > patience:
                loss_history.pop(0)

            # Check if the loss has not changed a lot over the past 'patience' iterations
            if len(loss_history) == patience and (max(loss_history) - min(loss_history) <= loss_change_threshold):
                print(
                    f"No major loss improvement in the last {patience} iterations. Stopping...")
                break

            if i >= 299:
                print(f"Maximum amount of iteration reached. Stopping at iteration {i + 1}")
                break

        prev_loss = loss.item()

        # Get the projection of the intersection where optimal z_val_tensor
        _, n_mask_torch = optimization_z_torch(z_val_tensor, scaled_mesh_pytorch3d, intrinsics_torch, edge_image_torch,
                                               H_C_O_mod_torch, H_I_C_torch, H_C_O_torch, img_mask_torch)
        if n_mask_torch is None:
            print('Mask is None')
            continue
        n_mask = (n_mask_torch.detach().numpy()).astype(int)
        img[n_mask[:, 0], n_mask[:, 1]] = [0, 0, 255]

        # Show results
        if debug:
            cv2.imshow("Image overlay", img)
            cv2.waitKey(0)

            cv2.imshow("Edge image", edge_image)
            cv2.waitKey(0)

            cv2.destroyAllWindows()

        # Show model TODO
        if show_model:
            full_model(z_val_tensor, scaled_mesh_pytorch3d, H_C_O_mod_torch, H_I_C_torch, H_C_O_torch)

        # Save the result
        if save:
            print("saved")
            png_img = (cv2.cvtColor(img, cv2.COLOR_BGR2BGRA))
            cv2.imwrite(os.path.join("output_img", f"{object_name}{rgb_idx + 1}.png"), png_img)

            # save overlayed version
            liquidmask_path = os.path.join("output_img", object_name+ "_new", f"prediction{rgb_idx + 1}.png")
            liquidmask = cv2.imread(liquidmask_path)
            if liquidmask is not None:
                alpha = 0.5
                where_mask = (liquidmask != [0, 0, 0]).any(axis=-1)
                liquidmask[where_mask] = [255,0,0] # make blue
                overlay = np.where(where_mask[..., None], (img * (1 - alpha) + liquidmask * alpha).astype(np.uint8), img)
                overlay[n_mask[:, 0], n_mask[:, 1]] = [0, 0, 255]
                png_overlay = (cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA))
                cv2.imwrite(os.path.join("output_img", f"{object_name}overlay{rgb_idx + 1}.png"), png_overlay)

            # save text file with optimal z_val
            # Define the text file path
            text_file_path = os.path.join("output_img", f"{object_name}.txt")

            # Convert line format
            new_line = f"{rgb_idx + 1}: {round(z_val_tensor.item(), 5)}\n"

            if os.path.exists(text_file_path):
                # If file exists, read lines
                with open(text_file_path, "r") as f:
                    lines = f.readlines()
            else:
                # If file doesn't exist, initialize empty list
                lines = []

                # Ensure the list has enough lines
            while len(lines) <= rgb_idx:
                lines.append("\n")  # Add empty lines if necessary

                # Modify the correct line
            lines[rgb_idx] = new_line

            # Write back the modified content
            with open(text_file_path, "w") as f:
                f.writelines(lines)

