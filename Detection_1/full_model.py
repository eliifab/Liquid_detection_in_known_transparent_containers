import os

import torch
import trimesh
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import plot_scene, plot_batch_individually
from pytorch3d.structures import Pointclouds
from scipy.spatial import Delaunay
from pytorch3d.structures import join_meshes_as_batch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

debug = False
save = False


def full_model(z_val: torch.Tensor, scaled_mesh: Meshes, H_C_O_mod: torch.Tensor, H_I_C: torch.Tensor,
               H_C_O: torch.Tensor) -> (torch.Tensor, torch.Tensor):

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

    # ************************************* Plane Mesh Intersection *************************************

    # Transform a plane normal depending on the new coordinate system
    R_plane = H_O_P[:3, :3]
    plane_normal = torch.tensor([0.0, 0.0, 1.0], device=z_val.device, dtype=z_val.dtype, requires_grad=True)
    plane_normal_new = torch.matmul(R_plane, plane_normal)
    plane_normal_new = plane_normal_new / torch.linalg.norm(plane_normal_new)

    # Get intersection of plane and mesh in object coordinates
    intersection_mesh = mesh_plane_clip(scaled_mesh, plane_normal_new, point_O_P)
    if intersection_mesh is None:
        return
    # TODO Handle case where no intersection occurs
    # Create figure and 3D axis
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    verts_intersection = intersection_mesh.verts_packed().detach().numpy()
    faces_intersection = intersection_mesh.faces_packed().detach().numpy()

    verts_object_part = scaled_mesh.verts_packed().detach().numpy()
    faces_object_part = scaled_mesh.faces_packed().detach().numpy()

    # Plot intersection mesh (Red, more visible)
    plot_mesh(ax, verts_intersection, faces_intersection, color="red", edgecolor='red', alpha=1)

    # Plot object part (Gray, semi-transparent)
    plot_mesh(ax, verts_object_part, faces_object_part, color="blue", edgecolor='blue', alpha=0.05)


    # Set limits
    ax.set_xlim(verts_object_part[:, 2].min(), verts_object_part[:, 2].max())
    ax.set_ylim(verts_object_part[:, 2].min(), verts_object_part[:, 2].max())
    ax.set_zlim(verts_object_part[:, 2].min(), verts_object_part[:, 2].max())

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
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio for all axes
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # ax.set_title("Mesh-Plane Intersection")
    ax.view_init(elev=20, azim=0)
    #plt.savefig(os.path.join("img_save", "MediumBottle_meshwithplane_46.png"), bbox_inches='tight', pad_inches=0)

    plt.show()

# Differentiable mesh-plane intersection by ChatGPT
def mesh_plane_clip(scaled_mesh, plane_normal, plane_origin):
    """
    todo
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

    # ------------------------------
    # Step 1: Compute the centroid of the points
    centroid = torch.mean(projections, dim=0)

    # Step 2: Sort points by angle around the centroid
    vectors = projections - centroid  # Vectors from centroid to points
    angles = torch.atan2(vectors[:, 1], vectors[:, 0])  # Angles in the XY plane
    sorted_indices = torch.argsort(angles)  # Sort points by angle
    sorted_points = projections[sorted_indices]

    # Step 3: Perform 2D triangulation on the sorted points
    points_2d = sorted_points[:, :2].detach().numpy()  # Use XY coordinates for 2D triangulation
    tri = Delaunay(points_2d)  # Triangulate the 2D points

    faces_new = torch.tensor(tri.simplices, dtype=torch.int64)

    # Step 4: Convert the faces to a tensor (PyTorch3D requires indices to be in tensor format)
    intersection_mesh = Meshes(verts=[projections],faces=[faces_new])  # Only filtered vertices

    return intersection_mesh

def plot_mesh(ax, vertices, faces, color, edgecolor, alpha=0.5):
    """Function to plot a triangular mesh"""
    mesh = Poly3DCollection(vertices[faces], alpha=alpha, edgecolor=edgecolor)
    mesh.set_facecolor(color)
    ax.add_collection3d(mesh)


