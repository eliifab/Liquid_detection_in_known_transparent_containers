import cv2
import torch
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pytorch3d.renderer import PerspectiveCameras, RasterizationSettings, PointLights, MeshRenderer, MeshRasterizer, \
    HardPhongShader
from pytorch3d.structures import Meshes
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import os

debug = False
debug_extra = False
debug_mesh = False
common_distance_loss = True
point_ray_loss = True


def optimization_z_torch(z_val: torch.Tensor, scaled_mesh: Meshes, intrinsics: torch.Tensor,
                         edge_image: torch.Tensor, H_C_O_mod: torch.Tensor, H_I_C: torch.Tensor,
                         H_C_O: torch.Tensor, img_mask: torch.Tensor) \
        -> (torch.Tensor, torch.Tensor):

    # ************************************* Transformations *************************************

    # Create [0,0,z_val,1] in a way the gradient can be tracked
    middlepoint_offset = torch.cat([torch.zeros(2, device=z_val.device, dtype=z_val.dtype), z_val[0:1],
                                    torch.tensor([1], device=z_val.device, dtype=z_val.dtype)], dim=0)

    # current z position (in object) in cam coord
    point_C_P = torch.matmul(H_C_O_mod, middlepoint_offset)

    # Get the vectors for the new system in cam coordinate system todo align z with gravitation if init system not aligned
    H_C_I = torch.linalg.inv(H_I_C)
    H_C_I_z = H_C_I[:3, 2]
    H_C_I_z = H_C_I_z / torch.linalg.norm(H_C_I_z)

    points_C_P_x = point_C_P[:3] / torch.linalg.norm(point_C_P[:3])
    points_C_P_y = torch.cross(H_C_I_z, points_C_P_x)
    points_C_P_y = points_C_P_y / torch.linalg.norm(points_C_P_y)

    points_C_P_x = torch.cross(points_C_P_y, H_C_I_z)
    points_C_P_x = points_C_P_x / torch.linalg.norm(points_C_P_x)


    # Rotation matrix of z position in cam coord
    R_C_P = torch.stack((points_C_P_x, points_C_P_y, H_C_I_z), dim=1)

    # Transformation from cam to new z position coord system
    H_C_P = torch.eye(4, device=z_val.device, dtype=z_val.dtype)
    H_C_P[:3, :3] = R_C_P
    H_C_P[:3, 3] = point_C_P[:3]

    # Transformation from object to new z position coord system
    H_O_P = torch.matmul(torch.linalg.inv(H_C_O), H_C_P)

    # Current z position in obj coord system
    point_O_P = torch.matmul(H_O_P, torch.tensor([0, 0, 0, 1], device=z_val.device, dtype=z_val.dtype))[:3]

    if debug:
        plot_mesh_with_dashed_z_axis(scaled_mesh, torch.linalg.inv(H_C_O)@H_C_O_mod, axis_length=0.2)

    if debug:
        # Create figure
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Draw coordinate frames
        draw_frame(ax, torch.eye(4), 'I', 'black',0.5)  # Initial system
        draw_frame(ax, H_I_C, 'C', 'orange',0.5)  # Camera coordinate system
        draw_frame(ax, H_I_C@H_C_P, 'P', 'magenta',0.5)  # Point system at optimal z

        # Set axis limits
        ax.set_xlim([-0.4, 0.7])
        ax.set_ylim([-0.7, 0.4])
        ax.set_zlim([-0.4, 0.7])
        ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio for all axes

        # Labels and grid
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        #ax.set_title("3D Visualization Initial, Camera, Point Coordinate Systems", fontsize=16)

        legend_coord = [Line2D([0], [0], color='black', marker='o', markersize=8, label='I (Initial)'),
                        Line2D([0], [0], color='orange', marker='o', markersize=8, label='C (Camera)'),
                        Line2D([0], [0], color='magenta', marker='o', markersize=8, label='P (Point)')]
        legend_axis = [Line2D([0], [0], color='red', marker='o', markersize=8, label='X Axis'),
                       Line2D([0], [0], color='green', marker='o', markersize=8, label='Y Axis'),
                       Line2D([0], [0], color='blue', marker='o', markersize=8, label='Z Axis')]

        # Show both legends
        ax.add_artist(plt.legend(handles=legend_coord, loc='lower left', title="Coordinate Systems"))
        ax.add_artist(plt.legend(handles=legend_axis, loc='lower right', title="Axes"))
        ax.view_init(elev=25, azim=25)
        #plt.savefig(os.path.join("img_save", "BigBottle_3coordsys_16.png"), bbox_inches='tight', pad_inches=0)

        plt.show()

    if debug:
        # Create figure
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Draw coordinate frames
        draw_frame(ax, H_I_C@H_C_O_mod, 'O', 'orange',0.5)  # Camera coordinate system
        draw_frame(ax, H_I_C@H_C_P, 'P', 'magenta',0.5)  # Point system at optimal z

        # Set axis limits
        ax.set_xlim([-0.5, 0.5])
        ax.set_ylim([-0.5, 0.5])
        ax.set_zlim([-0.5, 0.5])
        ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio for all axes

        # Labels and grid
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        #ax.set_title("3D Visualization Object, Point Coordinate Systems", fontsize=16)

        legend_coord = [Line2D([0], [0], color='orange', marker='o', markersize=8, label='O (Object)'),
                        Line2D([0], [0], color='magenta', marker='o', markersize=8, label='P (Point)')]
        legend_axis = [Line2D([0], [0], color='red', marker='o', markersize=8, label='X Axis'),
                       Line2D([0], [0], color='green', marker='o', markersize=8, label='Y Axis'),
                       Line2D([0], [0], color='blue', marker='o', markersize=8, label='Z Axis')]

        # Show both legends
        ax.add_artist(plt.legend(handles=legend_coord, loc='lower left', title="Coordinate Systems"))
        ax.add_artist(plt.legend(handles=legend_axis, loc='lower right', title="Axes"))
        plt.show()

    if debug:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        verts = scaled_mesh.verts_packed().detach().numpy()  # (N, 3) vertices for the mesh
        faces = scaled_mesh.faces_packed().detach().numpy()
        ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces, color='b', alpha=0.2)
        orig = point_O_P.detach()
        ax.scatter(orig[0], orig[1], orig[2], color='black', marker='x', s=100, linewidths=2, label="Current Z-Coordinate")

        # Draw the coordinate system at the origin
        draw_frame(ax, H_O_P, '', 'black',0.2)

        # Set axis limits
        ax.set_xlim([-0.2, 0.2])
        ax.set_ylim([-0.2, 0.2])
        ax.set_zlim([-0.2, 0.2])

        # Labels and grid
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        #ax.set_title("Point Coordinate System in Object", fontsize=16)
        ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio for all axes

        view_direction = -torch.linalg.inv(H_C_O_mod)[:3, 2].detach().numpy()
        azim = np.degrees(np.arctan2(view_direction[1], view_direction[0]))
        elev = np.degrees(np.arcsin(view_direction[2] / np.linalg.norm(view_direction)))
        ax.view_init(elev=elev, azim=azim)

        legend_axis = [Line2D([0], [0], color='red', marker='o', markersize=8, label='X Axis'),
                       Line2D([0], [0], color='green', marker='o', markersize=8, label='Y Axis'),
                       Line2D([0], [0], color='blue', marker='o', markersize=8, label='Z Axis'),
                       Line2D([0], [0], color='black', marker='x', markersize=10, label="Current Z-Coordinate")]

        ax.add_artist(plt.legend(handles=legend_axis, loc='upper right'))
        ax.view_init(elev=29, azim=-163)

        #plt.savefig(os.path.join("img_save", "BigBottle_bottlecoordsys_16.png"), bbox_inches='tight', pad_inches=0)
        plt.show()

    # ************************************* Plane Mesh Intersection *************************************

    # Transform a plane normal depending on the new coordinate system
    R_plane = H_O_P[:3, :3]
    plane_normal = torch.tensor([0.0, 0.0, 1.0], device=z_val.device, dtype=z_val.dtype, requires_grad=True)
    plane_normal_new = torch.matmul(R_plane, plane_normal)
    plane_normal_new = plane_normal_new / torch.linalg.norm(plane_normal_new)

    # Get intersection of plane and mesh in object coordinates
    intersection_points = mesh_plane_intersection(scaled_mesh, plane_normal_new, point_O_P)

    if intersection_points is None: # Handle case where no intersection occurs -> add current z value to intersection
        print('Outside of container')
        intersection_points = H_O_P[:3,3].view(1, 3)


    # to cam coord
    intersection_points_cam = torch.cat(
        [intersection_points, torch.ones(intersection_points.shape[0], 1, device=intersection_points.device,
                                         dtype=z_val.dtype)], dim=-1)
    intersection_points_cam = torch.matmul(H_C_O, intersection_points_cam.T).T[..., :3]

    # Project to 2D
    x_s = intersection_points_cam[:, 0] / intersection_points_cam[:, 2]
    y_s = intersection_points_cam[:, 1] / intersection_points_cam[:, 2]
    u = intrinsics[0, 0] * x_s + intrinsics[0, 2]
    v = intrinsics[1, 1] * y_s + intrinsics[1, 2]

    # if not entire obj in frame, clamp
    u_clamped = torch.round(u).long().clamp(0, edge_image.shape[1] - 1)
    v_clamped = torch.round(v).long().clamp(0, edge_image.shape[0] - 1)

    img_uv_mask = torch.zeros_like(edge_image, dtype=torch.float32)
    img_uv_mask[v_clamped, u_clamped] = 1.0

    coords_projection = torch.stack((v, u), dim=1)
    coords_projection_return = torch.stack((v_clamped, u_clamped), dim=1)


    # get coordinates where the mask != 0 -> in this area the projection will be kept
    coords_mask = torch.nonzero(img_mask, as_tuple=False).float()

    # get the points where the distance to any point in the mask is < 1 (coordinates are float -> can't index)
    diff = coords_projection[:, None, :] - coords_mask[None, :, :]
    dist_sq = torch.sum(diff ** 2, dim=-1)  # Squared distance (faster than sqrt)
    # boolean mask which projected vertices are in the object mask
    projection_cut_bool = torch.any(dist_sq < 1, dim=1)

    if projection_cut_bool.max() == False: # Handle case where no intersection occurs -> add current z value to intersection
        print('No points in object mask')

        z_val_nointersect = H_O_P[:4, 3].view(1, 4)
        z_val_nointersect_cam = torch.matmul(H_C_O, z_val_nointersect.T).T[..., :3]
        intersection_points_cam = z_val_nointersect_cam
        projection_cut_bool = torch.tensor([True])
        # Project to 2D
        z_x_s = z_val_nointersect_cam[:, 0] / z_val_nointersect_cam[:, 2]
        z_y_s = z_val_nointersect_cam[:, 1] / z_val_nointersect_cam[:, 2]
        z_u = intrinsics[0, 0] * z_x_s + intrinsics[0, 2]
        z_v = intrinsics[1, 1] * z_y_s + intrinsics[1, 2]
        coords_projection_cut = torch.stack((z_v, z_u), dim=1)

    else:
        coords_projection_cut = coords_projection[projection_cut_bool]

    coords_edge_prediction = torch.stack(torch.where(edge_image > 0), dim=1).float()

    if debug:
        plt.figure(figsize=(12, 12))
        plt.imshow(img_mask.detach().numpy(), cmap="gray")
        plt.scatter(coords_projection.detach().numpy()[:, 1], coords_projection.detach().numpy()[:, 0], color="red", s=1)
        plt.axis("off")
        #plt.savefig(os.path.join("img_save", "BigBottle_intersection2d_16.png"), bbox_inches='tight', pad_inches=0)
        plt.show()

    if debug:
        plt.figure(figsize=(12, 12))
        plt.imshow(img_mask.detach().numpy(), cmap="gray")
        plt.scatter(coords_projection_cut.detach().numpy()[:, 1], coords_projection_cut.detach().numpy()[:, 0], color="red", s=1)
        plt.axis("off")
        #plt.savefig(os.path.join("img_save", "BigBottle_intersection2d_cut_16.png"), bbox_inches='tight', pad_inches=0)
        plt.show()

    # *********************************************** Loss ***********************************************
    loss = 0

    # point_ray_loss: ------------------------------------------------------------------------------
    if point_ray_loss:
        x_cont, y_cont, z_cont = mask_to_rays(edge_image, intrinsics)
        ref_contour_rays = [torch.cat([x_cont[:, None], y_cont[:, None], z_cont[:, None]], dim=1).to(z_val.device)]
        contour_loss = torch.zeros(len(x_cont)).to(z_val.device)

        for ray_idx, rays in enumerate(ref_contour_rays):
            positions = intersection_points_cam[projection_cut_bool]
            t = torch.einsum(
                'ijkl,ijkl->ijk',
                rays[None,:,None,:],
                positions[:,None,None,:])

            pt_ray_distance = torch.norm(
                t*rays[None, ...] - positions[:,None,:],
                # p=2,
                dim=-1)

            closest_pt_to_rays = pt_ray_distance.min(dim=0).values.mean() # edge prediction to projection

            contour_loss[ray_idx] = (closest_pt_to_rays)*4

        loss = loss + torch.sum(contour_loss)

    if common_distance_loss:

        ################ get z direction in 2d ##################

        origin = H_C_O_mod @ torch.tensor([0,0,0,1], dtype=H_C_O_mod.dtype)
        direction = H_C_O_mod @ torch.tensor([0,0,0.05,1], dtype=H_C_O_mod.dtype)

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

        direction_uv = torch.stack((-direction_vv,direction_uu))
        object_z_direction = direction_uv / torch.norm(direction_uv)

        angle = torch.arctan2(object_z_direction[1], object_z_direction[0]) - torch.arctan2(torch.tensor(1.0), torch.tensor(0.0))
        # torch and numpy seem to have other x y conventions

        # Create the rotation matrix
        R_C_O_2d = torch.tensor([
            [torch.cos(angle), -torch.sin(angle)],
            [torch.sin(angle), torch.cos(angle)]
        ])
        object_y_direction = R_C_O_2d[:,0]

        diffs = coords_projection_cut[:, None, :] - coords_edge_prediction[None, :, :]  # Pairwise differences

        # directions in object z direction
        diff_rotated = torch.einsum('ij,...j->...i', R_C_O_2d, diffs)
        # Set a narrow bandwidth
        band_width = 1.0
        mask_valid = torch.where(torch.abs(diff_rotated[:,:,0]) <= band_width, 1, 0)  # Binary mask
        mask_not_valid = torch.where(torch.abs(diff_rotated[:,:,0]) > band_width, 1, 0)

        # --- Get the filtered points and their distances ---
        valid_distances = diff_rotated * mask_valid[..., None]  # Mask the diffs, keeping the shape (N, M, 2)
        valid_distances_z = valid_distances[:,:,1]

        # if no points there to be optimized
        if valid_distances_z.max() == 0 and valid_distances_z.min() == 0:
            return loss, coords_projection

        if debug or debug_extra:
            plt.figure(figsize=(6, 6))

            # Convert tensors to NumPy for visualization
            p1_np = coords_edge_prediction.detach().numpy()
            p2_np = coords_projection_cut.detach().numpy()

            # Find valid matches
            valid_matches = mask_valid > 0.5  # Since sigmoid outputs values between 0 and 1
            valid_p1 = coords_edge_prediction[valid_matches.any(dim=0)].detach().numpy()  # Get valid points from pointset2
            valid_p2 = coords_projection_cut[valid_matches.any(dim=1)].detach().numpy()  # Get valid points from pointset2

            # Plot all points
            plt.scatter(p1_np[:, 1], p1_np[:, 0], s=10, color='blue', label='Unmatched Predicted Points')
            plt.scatter(p2_np[:, 1], p2_np[:, 0], s=10, color='gray', alpha=0.5, label='Unmatched Projected Points')
            plt.scatter(valid_p1[:, 1], valid_p1[:, 0], s=10, color='red', label='Valid Matches')
            plt.scatter(valid_p2[:, 1], valid_p2[:, 0], s=10, color='red')

            #plt.scatter(origin_u, origin_v, s=10, color='purple')
            #plt.scatter(direction_u, direction_v, s=10, color='pink')

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
            plt.ylim(new_y_min, new_y_max)

            # Define bottom-center position
            start_point = (new_x_min + max_range/10 + 2, new_y_max - max_range/10 - 2)

            # Get the x and y components of the direction vectors
            y_direction_x, y_direction_y = object_y_direction[1].item(), object_y_direction[0].item()
            z_direction_x, z_direction_y = object_z_direction[1].item(), object_z_direction[0].item()
            plt.quiver(start_point[0], start_point[1], y_direction_x * max_range/10, -y_direction_y * max_range/10,
                       angles='xy', scale_units='xy', scale=1, color='green', label="Object Y Direction", linewidth=2)
            plt.quiver(start_point[0], start_point[1], z_direction_x * max_range/10, -z_direction_y * max_range/10,
                       angles='xy', scale_units='xy', scale=1, color='orange', label="Object Z Direction", linewidth=2)

            # Formatting
            plt.legend(loc='lower right', bbox_to_anchor=(1.6, 0))
            plt.xlabel("X")
            plt.ylabel("Y")
            #plt.title("Points in valid direction")
            plt.grid(True)
            plt.gca().invert_yaxis()  # Invert Y-axis if needed
            # Set fixed position (e.g., 300 pixels from left, 200 pixels from top)
            fixed_x = 300  # Horizontal position
            fixed_y = 200  # Vertical position
            fig_width = 800  # Window width
            fig_height = 800  # Window height

            # Move the figure window to the fixed position
            mng = plt.get_current_fig_manager()
            mng.window.wm_geometry(f"{fig_width}x{fig_height}+{fixed_x}+{fixed_y}")
            #plt.savefig(os.path.join("img_save", "MediumBottle_46_valid_direction.png"), bbox_inches='tight', pad_inches=0.1)
            plt.show()

        # Find the most common distances (within a certain margin)
        valid_distances_z_flat = valid_distances_z[mask_not_valid != 1].flatten()

        if len(valid_distances_z) != 0:

            sorted_distances_z, _ = torch.sort(valid_distances_z_flat)
            threshold = 1  # Distance window # todo variable
            # Use binary search to find valid neighbors

            left_indices = torch.searchsorted(sorted_distances_z, sorted_distances_z - threshold, side="left")
            right_indices = torch.searchsorted(sorted_distances_z, sorted_distances_z + threshold, side="right")

            # Compute the number of neighbors within the threshold
            counts = right_indices - left_indices
            most_common_index = torch.argmax(counts)
            most_common_distance = sorted_distances_z[most_common_index]

            within_range = (valid_distances_z >= most_common_distance - threshold) & (
                    valid_distances_z <= most_common_distance + threshold ) & (mask_not_valid != 1)

            if debug_extra or debug:
                plt.figure(figsize=(6, 6))
                # Convert tensors to NumPy for visualization
                p1_np = coords_edge_prediction.detach().numpy()
                p2_np = coords_projection_cut.detach().numpy()
                # Mask valid matches
                filtered_p1 = coords_edge_prediction[within_range.any(dim=0)].detach().numpy()  # Get only valid matches
                filtered_p2 = coords_projection_cut[within_range.any(dim=1)].detach().numpy()  # Get only valid matches

                # Plot all points
                plt.scatter(p1_np[:, 1], p1_np[:, 0], s=10, color='blue', label='Unmatched Predicted Points')
                plt.scatter(p2_np[:, 1], p2_np[:, 0], s=10, color='gray', alpha=0.5, label='Unmatched Projected Points')
                plt.scatter(filtered_p1[:, 1], filtered_p1[:, 0], s=10, color='red', label='Filtered Matches')
                plt.scatter(filtered_p2[:, 1], filtered_p2[:, 0], s=10, color='red')

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
                plt.ylim(new_y_min, new_y_max)

                # Define bottom-center position
                start_point = (new_x_min + max_range / 10 + 2, new_y_max - max_range / 10 - 2)

                # Get the x and y components of the direction vectors
                y_direction_x, y_direction_y = object_y_direction[1].item(), -object_y_direction[0].item()
                z_direction_x, z_direction_y = object_z_direction[1].item(), -object_z_direction[0].item()
                plt.quiver(start_point[0], start_point[1], y_direction_x * max_range / 10,
                           y_direction_y * max_range / 10,
                           angles='xy', scale_units='xy', scale=1, color='green', label="Object Y Direction",
                           linewidth=2)
                plt.quiver(start_point[0], start_point[1], z_direction_x * max_range / 10,
                           z_direction_y * max_range / 10,
                           angles='xy', scale_units='xy', scale=1, color='orange', label="Object Z Direction",
                           linewidth=2)

                plt.legend(loc='lower right', bbox_to_anchor=(1.6, 0))

                plt.xlabel("X")
                plt.ylabel("Y")
                #plt.title("Points most common distance")
                plt.grid(True)
                plt.gca().invert_yaxis()

                # Set fixed position (e.g., 300 pixels from left, 200 pixels from top)
                fixed_x = 300  # Horizontal position
                fixed_y = 200  # Vertical position
                fig_width = 800  # Window width
                fig_height = 800  # Window height

                # Move the figure window to the fixed position
                mng = plt.get_current_fig_manager()
                mng.window.wm_geometry(f"{fig_width}x{fig_height}+{fixed_x}+{fixed_y}")
                #plt.savefig(os.path.join("img_save", "MediumBottle_46_common_distance.png"), bbox_inches='tight', pad_inches=0.1)
                plt.show()

            loss = loss + torch.abs(torch.mean(valid_distances[within_range]))*0.01
            loss = loss/2


    return loss, coords_projection_return


# **********************************************************************************************************

def mask_to_rays(mask, intrinsics, normalize=True):
    indices_v, indices_u = torch.nonzero(mask, as_tuple=True)
    x = (indices_u - intrinsics[0, 2]) / intrinsics[0, 0]
    y = (indices_v - intrinsics[1, 2]) / intrinsics[1, 1]
    z = torch.ones(x.shape, device=mask.device)

    if normalize:
        l2_norm = torch.sqrt(x ** 2 + y ** 2 + 1.)
        x /= l2_norm
        y /= l2_norm
        z /= l2_norm

    return x, y, z

def print_grad(grad):
    print("Gradient: ", grad)  # usage: variable.register_hook(print_grad) to see gradients for debugging


# Differentiable mesh-plane intersection by ChatGPT
def mesh_plane_intersection(scaled_mesh, plane_normal, plane_origin):
    """
    Differentiable computation of the intersection of a mesh with a plane.

    Args:
        scaled_mesh (Meshes): A PyTorch3D Meshes object representing the mesh.
        plane_normal (torch.Tensor): The normal vector of the slicing plane (3,).
        plane_origin (torch.Tensor): A point on the slicing plane (3,).
    Returns:
        torch.Tensor: Intersection points as a tensor of shape (N, 3), where N is the number of intersection points.
    """

    # Extract vertices and faces from the mesh
    vertices = scaled_mesh.verts_padded()  # Shape (1, V, 3)
    faces = scaled_mesh.faces_padded()  # Shape (1, F, 3)

    # Compute signed distances of vertices from the plane
    # distances = (vertices - plane_origin) * plane_normal
    signed_distances = torch.einsum('bvj,j->bv', vertices - plane_origin, plane_normal)  # Shape (1, V)

    # Gather face vertices and distances
    face_vertices = vertices[:, faces]  # Shape (1, F, 3, 3)
    face_distances = signed_distances[:, faces]  # Shape (1, F, 3)

    # Compute intersections for all edges in a vectorized manner
    i_indices = torch.tensor([0, 1, 2], device=vertices.device)
    j_indices = torch.tensor([1, 2, 0], device=vertices.device)

    d_i, d_j = face_distances[..., i_indices], face_distances[..., j_indices]  # Shape (1, F, 3)
    v_i, v_j = face_vertices[..., i_indices, :], face_vertices[..., j_indices, :]  # Shape (1, F, 3, 3)

    # Identify intersecting edges (sign change)
    mask = (d_i * d_j) < 0  # Shape (1, F, 3)

    # Compute intersection factors t = d_i / (d_i - d_j)
    denom = d_i - d_j
    epsilon = 1e-6
    denom = torch.where(torch.abs(denom) < epsilon, torch.ones_like(denom), denom)  # Avoid division by zero
    t = d_i / denom  # Shape (1, F, 3)

    # Compute intersection points
    intersection_points = v_i + t.unsqueeze(-1) * (v_j - v_i)  # Shape (1, F, 3, 3)

    # Mask invalid intersections
    valid_intersections = intersection_points[mask]  # Shape (N, 3)

    if valid_intersections.numel() == 0:
        return None

    # Project intersection points exactly onto the plane
    plane_normal = plane_normal / torch.norm(plane_normal)  # Normalize the normal
    to_plane = valid_intersections - plane_origin
    distances = torch.sum(to_plane * plane_normal, dim=-1, keepdim=True)  # Signed distance
    projections = valid_intersections - distances * plane_normal  # Project to plane
    if debug or debug_mesh:
        visualize_intersection(scaled_mesh, projections, plane_normal, plane_origin)

    # todo place points evenly in 2d

    return projections


def visualize_intersection(scaled_mesh, intersection_points, plane_normal, plane_origin):
    vertices = scaled_mesh.verts_packed().detach().numpy()
    faces = scaled_mesh.faces_packed().detach().numpy()
    intersection_points = intersection_points.detach().numpy()
    plane_normal = plane_normal.detach().numpy()
    plane_origin = plane_origin.detach().numpy()

    # Create figure and 3D axis
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection="3d")

    # Plot mesh surface
    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=faces, color='b', alpha=0.2)

    # Plot intersection points
    ax.scatter(intersection_points[:, 0], intersection_points[:, 1], intersection_points[:, 2],
               color="red", s=1, label="Intersection Points")
    # Plot plane normal vector (starting at plane_origin)
    ax.quiver(plane_origin[0], plane_origin[1], plane_origin[2],
              plane_normal[0], plane_normal[1], plane_normal[2],
              color='black', length=0.05, linewidth=2, label="Plane Normal", arrow_length_ratio=0.2)

    ax.scatter(plane_origin[0], plane_origin[1], plane_origin[2],
               color="black", s=100, marker="x", label="Plane Origin")  # Large point for visibility

    # Remove all axis elements
    ax.axis("off")  # Removes all axes, labels, and grid

    # Set labels
    ax.set_facecolor('none')
    ax.xaxis.pane.set_visible(False)
    ax.yaxis.pane.set_visible(False)
    ax.zaxis.pane.set_visible(False)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.set_xlim([vertices[:, 2].min() - 0.0, vertices[:, 2].max() + 0.0])
    ax.set_ylim([vertices[:, 2].min() - 0.0, vertices[:, 2].max() + 0.0])
    ax.set_zlim([vertices[:, 2].min() - 0.0, vertices[:, 2].max() + 0.0])
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio for all axes
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    #ax.set_title("Mesh-Plane Intersection")
    ax.view_init(elev=30, azim=-100)
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label="Intersection Points"),
        Line2D([0], [0], marker='x', color='black', markerfacecolor='black', markersize=10, label="Plane Origin"),
        Line2D([0], [0], marker=(3, 0, 0), color='black', markersize=10, label="Plane Normal")  # Arrow symbol
    ]
    plt.legend(handles= legend_elements, loc="center right", fontsize=17)
    #plt.savefig(os.path.join("img_save", "BigBottle_meshplane_intersect_16.png"), bbox_inches='tight', pad_inches=0)
    plt.show()

# visualize debug - chatgpt
def draw_frame(ax, H, label, color, axis_length):
    """Draw a coordinate frame given a 4x4 transformation matrix H (torch tensor)."""
    origin = H[:3, 3].detach().numpy()  # Extract translation (position)
    R = H[:3, :3].detach().numpy()  # Extract rotation matrix

    # Compute endpoints for axes
    x_axis = R @ np.array([axis_length, 0, 0]) + origin
    y_axis = R @ np.array([0, axis_length, 0]) + origin
    z_axis = R @ np.array([0, 0, axis_length]) + origin

    # Draw axes
    ax.quiver(*origin, *(x_axis - origin), color='r', label=f'{label} X' if label else None)
    ax.quiver(*origin, *(y_axis - origin), color='g', label=f'{label} Y' if label else None)
    ax.quiver(*origin, *(z_axis - origin), color='b', label=f'{label} Z' if label else None)

    # Label the origin
    ax.text(*origin, f"{label}", color=color, fontsize=20, fontweight='bold')

# visualize debug - chatgpt
def plot_mesh_with_dashed_z_axis(scaled_mesh, H, axis_length=1.0):
    """
    Plots a mesh with only the Z-axis of a given coordinate system.
    The Z-axis is dashed and extends fully through the mesh with arrowheads in the middle.
    """
    # Extract vertices and faces from the scaled_mesh

    verts = scaled_mesh.verts_packed().detach().numpy()
    faces = scaled_mesh.faces_packed().detach().numpy()

    # Extract the Z-axis from transformation matrix H (it’s the third column of the rotation part)
    origin = H[:3, 3].detach().numpy()  # Translation (origin of the new system)
    z_axis = H[:3, 2].detach().numpy()  # Z-axis direction

    # Normalize Z-axis and scale to match the desired length
    z_axis = z_axis / np.linalg.norm(z_axis) * axis_length

    # Define start and end points for the dashed Z-axis
    z_start = origin - z_axis  # Extend below the object
    z_end = origin + z_axis    # Extend above the object

    # Compute the middle point for the arrows
    z_mid = (z_start + z_end) / 2

    # Create figure and axis
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the mesh
    ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces, color='b', alpha=0.2)

    # Plot the dashed Z-axis
    ax.plot([z_start[0], z_end[0]], [z_start[1], z_end[1]], [z_start[2], z_end[2]],
            linestyle="dashed", color="black", linewidth=2)

    # Compute arrow directions (scaled unit vector of Z-axis)
    arrow_length = 0.03  # Adjust as needed
    z_arrow_up = arrow_length * (z_axis / np.linalg.norm(z_axis))
    z_arrow_down = -z_arrow_up  # Opposite direction

    # Add arrowheads in the correct direction
    ax.quiver(z_mid[0], z_mid[1], z_mid[2],
              z_arrow_up[0], z_arrow_up[1], z_arrow_up[2],
              color="red", arrow_length_ratio=0.5, linewidth=2)

    ax.quiver(z_mid[0], z_mid[1], z_mid[2],
              z_arrow_down[0], z_arrow_down[1], z_arrow_down[2],
              color="red", arrow_length_ratio=0.5, linewidth=2)

    # Remove background, axes, and grid
    ax.set_facecolor('none')
    ax.xaxis.pane.set_visible(False)
    ax.yaxis.pane.set_visible(False)
    ax.zaxis.pane.set_visible(False)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.set_xlim([verts[:, 2].min() - 0.1, verts[:, 2].max() + 0.1])
    ax.set_ylim([verts[:, 2].min() - 0.1, verts[:, 2].max() + 0.1])
    ax.set_zlim([verts[:, 2].min() - 0.1, verts[:, 2].max() + 0.1])
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio for all axes

    ax.set_title("Optimization Z Visualization", fontsize=16)

    # Show the plot
    plt.show()