#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor
import os
from matplotlib.lines import Line2D


load_prediction = False  # if prediction already is pre-saved
debug = False  # output image
debug_extra = False
fine_debug = False  # also in between images
save = False


def img_edge_prediction(img: np.array, img_mask: np.array, rgb_idx: int, object_name: str,
                        H_I_C: np.array, H_C_O_mod: np.array, intrinsics: np.array, kernel_width: int, kernel_width2: int) -> np.array:
    # segment anything

    origin = H_C_O_mod @ np.array([0, 0, 0, 1])
    H_O_I_mod = np.linalg.inv(H_C_O_mod) @ np.linalg.inv(H_I_C)
    H_O_I_z = H_O_I_mod[:, 2]
    direction = H_C_O_mod @ (np.array([0, 0, 0, 1]) + 0.05 * H_O_I_z) # point in gravity direction

    origin_x = origin[0] / origin[2]
    origin_y = origin[1] / origin[2]
    origin_u = intrinsics[0, 0] * origin_x + intrinsics[0, 2]
    origin_v = intrinsics[1, 1] * origin_y + intrinsics[1, 2]

    direction_x = direction[0] / direction[2]
    direction_y = direction[1] / direction[2]
    direction_u = intrinsics[0, 0] * direction_x + intrinsics[0, 2]
    direction_v = intrinsics[1, 1] * direction_y + intrinsics[1, 2]

    direction_uu = direction_u - origin_u
    direction_vv = direction_v - origin_v

    direction_uv = np.stack((-direction_vv, direction_uu))  # minus due to conventions
    gravity_z_direction = direction_uv / np.linalg.norm(direction_uv)
    #gravity_z_direction = - gravity_z_direction # p_020
    angle = np.arctan2(gravity_z_direction[1], gravity_z_direction[0]) - np.arctan2(0, 1)

    # Create the rotation matrix to rotate back
    R_G_C_2d = np.array([
        [np.cos(-angle), -np.sin(-angle)],
        [np.sin(-angle), np.cos(-angle)]
    ])

    if fine_debug:
        plt.figure(figsize=(6, 6))
        plt.imshow(img[:, :, [2, 1, 0]], extent=[0, img.shape[1], img.shape[0], 0])

        # set limits to always have the same ratio for x and y axis

        # Get the x and y components of the direction vector
        z_direction_x, z_direction_y = gravity_z_direction[1].item(), gravity_z_direction[0].item()
        plt.quiver(origin_u, origin_v, z_direction_x * 100, -z_direction_y * 100,
                   angles='xy', scale_units='xy', scale=1, color='red', label="Gravity Direction", linewidth=1)

        #plt.scatter(origin_u, origin_v, color='blue', label='org', marker='x')
        #plt.scatter(direction_u, direction_v, color='pink', label='dir', marker='x')

        # Formatting
        plt.legend(loc="upper right")
        #plt.xlabel("X")
        #plt.ylabel("Y")
        plt.axis('off')
        #plt.title("Points in valid direction")
        plt.grid(False)
        #plt.gca().invert_yaxis()  # Invert Y-axis if needed
        # Set fixed position (e.g., 300 pixels from left, 200 pixels from top)
        fixed_x = 300  # Horizontal position
        fixed_y = 200  # Vertical position
        fig_width = 800  # Window width
        fig_height = 800  # Window height

        # Move the figure window to the fixed position
        mng = plt.get_current_fig_manager()
        mng.window.wm_geometry(f"{fig_width}x{fig_height}+{fixed_x}+{fixed_y}")
        #plt.savefig(os.path.join("img_save", "BigBottle_91_gravityvector.png"), bbox_inches='tight', pad_inches=0)

        plt.show()

    # make mask smaller so that points don't come to lay on borderpixels
    mod_mask = (img_mask / np.max(img_mask)).astype(np.uint8)
    mod_mask = cv2.filter2D(mod_mask, kernel=np.ones((kernel_width2, kernel_width2)), ddepth=-1)
    mod_mask = np.where(mod_mask != kernel_width2 ** 2,0,1)
    mask_coord_rot = (R_G_C_2d @ np.array(np.where(mod_mask))[::-1]).T

    if False: # only for debug MediumBottle 5
        plt.figure(figsize=(10, 10))
        new_mask = np.zeros((img_mask.shape[0],img_mask.shape[1],3)).astype(np.uint8)
        new_mask[np.where(img_mask==0)] = (255,255,255)
        new_mask[np.where(img_mask!=0)] = (0,0,255)
        new_mask[np.where(mod_mask!=0)] = (255,0,0)

        plt.imshow(new_mask[np.where(new_mask[:,:,2] == 0)[0].min()-30:np.where(new_mask[:,:,2] == 0)[0].max()+50,
                   np.where(new_mask[:,:,2] == 0)[1].min()-200:np.where(new_mask[:,:,2] == 0)[1].max()+200,:])

        # Optional: Remove axes for clarity
        plt.axis('off')
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Original Image Mask'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Shrunken Image Mask')
        ]

        # Add the legend manually to the plot
        plt.legend(handles=legend_elements, loc='upper left',fontsize=17)
        # Show the plot
        #plt.savefig(os.path.join("img_save", "MediumBottle_5_shrunken_mask.png"), bbox_inches='tight', pad_inches=0)
        plt.show()

    if mask_coord_rot.shape[0] == 0:
        print('Not enough points in mask for liquid prediction')
        liquidmask = (np.zeros_like(img_mask)).astype(np.uint8)
        if save or not load_prediction:
            png_img = (cv2.cvtColor(liquidmask, cv2.COLOR_BGR2BGRA))

            output_path = os.path.join("output_img", object_name)
            os.makedirs(output_path, exist_ok=True)  # Create folder if it doesn't exist
            cv2.imwrite(os.path.join(output_path, f"prediction{rgb_idx + 1}.png"), png_img)

        return liquidmask  # return original mask

    z_min = mask_coord_rot[:, 1].min()
    z_max = mask_coord_rot[:, 1].max()
    z_height = z_max-z_min

    # define two points on the bottle, one in the liquid and one outside for SAM

    if object_name == 'BigBottle': # big & med 0.3 & 0.8 for p_020
        high_row_select = 0.2 * z_height + z_min
    elif object_name == 'MediumBottle':
        high_row_select = 0.2 * z_height + z_min
    elif object_name == 'SmallBottle':
        high_row_select = 0.2 * z_height + z_min

    if object_name == 'BigBottle':
        low_row_select = 0.85 * z_height + z_min
    elif object_name == 'MediumBottle':
        low_row_select = 0.95 * z_height + z_min
    elif object_name == 'SmallBottle':
        low_row_select = 0.9 * z_height + z_min

    high_row_select = min(mask_coord_rot[:, 1], key=lambda x: abs(x - high_row_select))
    high_row_columns = np.where((mask_coord_rot[:, 1] > np.floor(high_row_select)-1) &
                               (mask_coord_rot[:, 1] < np.ceil(high_row_select)+1))[0]
    high_column_medium = mask_coord_rot[high_row_columns][:, 0].min() + (
                mask_coord_rot[high_row_columns][:, 0].max() - mask_coord_rot[high_row_columns][:, 0].min()) / 2
    high_column_select = min(mask_coord_rot[high_row_columns][:,0], key=lambda x: abs(x - high_column_medium))

    low_row_select = min(mask_coord_rot[:, 1], key=lambda x: abs(x - low_row_select))
    low_row_columns = np.where((mask_coord_rot[:, 1] > np.floor(low_row_select)-1) &
                              (mask_coord_rot[:, 1] < np.ceil(low_row_select)+1))[0]
    low_column_medium = mask_coord_rot[low_row_columns][:, 0].min() + (
                mask_coord_rot[low_row_columns][:, 0].max() - mask_coord_rot[low_row_columns][:, 0].min()) / 2
    low_column_select = min(mask_coord_rot[low_row_columns][:, 0], key=lambda x: abs(x - low_column_medium))

    high_point = (np.linalg.inv(R_G_C_2d) @ np.stack((high_column_select, high_row_select)))[::-1]
    low_point = (np.linalg.inv(R_G_C_2d) @ np.stack((low_column_select, low_row_select)))[::-1]

    if debug or fine_debug:
        plt.figure(figsize=(6, 6))
        plt.scatter(np.array(np.where(img_mask))[::-1].T[:, 0], np.array(np.where(img_mask))[::-1].T[:, 1], color='blue', label='Original Points', marker='.')
        plt.scatter(mask_coord_rot[:, 0], mask_coord_rot[:, 1], color='red', label='Rotated Points', marker='.')
        plt.scatter(high_column_select, high_row_select, s=100, color='pink', label='High Point', marker='x')
        plt.scatter(low_column_select, low_row_select, s=100, color='green', label='Low Point', marker='x')
        # set limits to always have the same ratio for x and y axis
        x_min, x_max = plt.xlim()
        y_min, y_max = plt.ylim()
        x_range = x_max - x_min
        y_range = y_max - y_min
        max_range = max(x_range, y_range)
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        new_x_min = x_center - max_range / 2
        new_x_max = x_center + max_range / 2
        new_y_min = y_center - max_range / 2
        new_y_max = y_center + max_range / 2
        plt.xlim(new_x_min, new_x_max)
        plt.ylim(new_y_max, new_y_min)

        # Define bottom-center position
        start_point = (new_x_min + max_range / 10 + 2, new_y_max - max_range / 10 - 2)

        # Get the x and y components of the direction vector
        z_direction_x, z_direction_y = gravity_z_direction[1].item(), gravity_z_direction[0].item()
        plt.quiver(start_point[0], start_point[1], z_direction_x * 100, -z_direction_y * 100,
                   angles='xy', scale_units='xy', scale=1, color='blue', label="Original Gravity vector", linewidth=2)

        plt.quiver(start_point[0], start_point[1], 0, -1 * 100,
                   angles='xy', scale_units='xy', scale=1, color='red', label="Rotated Gravity vector", linewidth=2)

        # Formatting
        plt.legend(loc='lower right', bbox_to_anchor=(1.05, 0))
        # Add labels and legend
        plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        #plt.title("Original vs. Rotated Points")
        plt.grid(False)
        #plt.savefig(os.path.join("img_save", "points_BigBottle_91_rotated_mask.png"), bbox_inches='tight', pad_inches=0.1)
        plt.show()

    # test end

    # extract the 2d object
    img_mask = (img_mask / np.max(img_mask)).astype(np.uint8)  # mask to gray

    input_point = np.array([[low_point[1], low_point[0]],[high_point[1], high_point[0]]])

    # define the labels (1 = inside mask, 0 = outside mask)
    input_label = np.array([1, 0])

    if fine_debug:  # show points on image
        plt.figure(figsize=(10, 10))
        plt.imshow(img[..., ::-1]*img_mask[..., None])
        show_points(input_point, input_label, plt.gca())
        plt.axis('off')
        #plt.savefig(os.path.join("img_save", "points_BigBottle_1_mask.png"), bbox_inches='tight', pad_inches=0)
        plt.show()

    if fine_debug:  # show points on image
        plt.figure(figsize=(10, 10))
        plt.imshow(img[..., ::-1])
        show_points(input_point, input_label, plt.gca())
        plt.axis('off')
        #plt.savefig(os.path.join("img_save", "points_MediumBottle_19_full.png"), bbox_inches='tight', pad_inches=0)
        plt.show()

    # since predicting the mask takes some time, loading pre-saved predictions is faster
    if load_prediction:
        file_path = os.path.join("output_img", object_name, f"prediction{rgb_idx + 1}.png")
        liquidmask = cv2.imread(file_path)

    else:
        # set up the predictor
        print('Start SAM')
        sam_checkpoint = "sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        device = "cpu"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        predictor = SamPredictor(sam)
        predictor.set_image(img)

        # predict the liquid mask
        liquidmask, _, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )
        liquidmask = (np.squeeze(liquidmask) * 255).astype(dtype=np.uint8)
        print('SAM done')

    if debug or fine_debug:
        cv2.imshow("liquidmask", liquidmask)
        cv2.waitKey()

    if save or not load_prediction:
        png_img = (cv2.cvtColor(liquidmask, cv2.COLOR_BGR2BGRA))

        output_path = os.path.join("output_img", object_name)
        os.makedirs(output_path, exist_ok=True)  # Create folder if it doesn't exist
        cv2.imwrite(os.path.join(output_path, f"prediction{rgb_idx + 1}.png"), png_img)

    # -------------------------------------------postprocess the mask-------------------------------------------

    if liquidmask is None:
        print('Please generate the masks first')
        return np.zeros_like(img_mask)


    kernel = np.ones((5,5), np.uint8)  # Increase size if needed

    # Apply morphological closing to fill holes
    liquidmask_morph = cv2.morphologyEx(liquidmask, cv2.MORPH_CLOSE, kernel, iterations=5)

    if fine_debug:
        cv2.imshow("liquidmask", liquidmask_morph)
        cv2.waitKey()

    if fine_debug:

        # Define an overlay color for the morphed mask (e.g., red)
        overlay_color = np.array([150, 150, 150], dtype=np.uint8)  # Red color

        # Create a colored overlay where liquidmask_morph is active
        mask_overlay = np.where(liquidmask_morph > 0, overlay_color, 0).astype(np.uint8)

        # Blend the original mask and the overlay with some transparency
        alpha = 0.9  # Adjust transparency
        overlayed_image = cv2.addWeighted(liquidmask, 1, mask_overlay, alpha, 0)

        # Show the result
        cv2.imshow("Overlayed Mask", overlayed_image)
        cv2.waitKey(0)

    # get the edges of the mask
    edge_image = cv2.Canny(liquidmask_morph, 100, 200)
    if fine_debug:
        cv2.imshow("edge img", edge_image)
        cv2.waitKey()

    # modify the mask to remove edges close to the object border and only keep the liquid related one
    mod_mask = cv2.filter2D(img_mask, kernel=np.ones((kernel_width, kernel_width)), ddepth=-1)
    mod_mask[np.where(mod_mask != kernel_width ** 2)] = 0
    edge_image = np.logical_and(edge_image, mod_mask) * edge_image
    if fine_debug:
        cv2.imshow("borders removed edge img", edge_image)
        cv2.waitKey()

    bitwise_img = np.zeros_like(edge_image)
    kernel = np.ones((3, 3), np.uint8)
    edge_image = cv2.morphologyEx(edge_image, cv2.MORPH_CLOSE, kernel)  # to connect edges that have been separated
    nb_labels, label_im = cv2.connectedComponents(edge_image, connectivity=8)
    unique_labels, indices, counts = np.unique(label_im, return_index=True, return_counts=True)
    if len(unique_labels) != 1:
        # only keep the edges that are longer than 0.5 times the longest edge to remove noise
        max_count = np.max(counts[1:])
        valid_labels = unique_labels[counts > max_count / 2][1:]

        bitwise_img[np.isin(label_im, valid_labels)] = 1
        bitwise_img *= edge_image

    if debug or fine_debug:
        cv2.imshow("Filtered edges img", bitwise_img)
        cv2.waitKey()

    if debug or fine_debug:
        img[bitwise_img.astype(bool)] = [0,0,255]
        cv2.imshow("Filtered edges img", img)
        cv2.waitKey()

    return bitwise_img


# -------------------------------------------for debug-------------------------------------------

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
