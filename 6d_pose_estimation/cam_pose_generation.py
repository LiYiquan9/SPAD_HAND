import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import os
import json

def read_real_data(real_dataset_path):
    with open(os.path.join(real_dataset_path, "001", "tmf.json")) as f:
        tmf_data = json.load(f)
    poses_homog = np.array([measurement["pose"] for measurement in tmf_data])  # Shape: (N, 4, 4)
    return poses_homog

def save_camera_poses(poses, filename="camera_poses.npz"):
    """
    Save generated camera poses to an npz file.

    Args:
        poses (list of np.ndarray): List of 4x4 transformation matrices (camera-to-world).
        filename (str): Path to save the .npz file.
    """
    poses_array = np.array(poses)  # Convert list to numpy array (N, 4, 4)
    np.savez_compressed(filename, poses=poses_array)
    print(f"Saved {len(poses)} camera poses to {filename}")
    
def generate_camera_poses(num_cameras, radius=0.4):
    """
    Generate camera poses (rotation and translation) uniformly distributed on a semi-sphere.

    Args:
        num_cameras (int): Number of camera poses to generate.
        radius (float): Radius of the hemisphere.

    Returns:
        poses (list of np.ndarray): List of 4x4 transformation matrices (camera-to-world).
    """
    # Sample uniform points on a hemisphere using spherical coordinates
    theta = np.random.uniform(0, 2 * np.pi, num_cameras)  # Azimuth angle
    phi = np.random.uniform(0, np.pi / 2, num_cameras)  # Elevation angle (semi-sphere)

    # Convert to Cartesian coordinates
    x = radius * np.cos(theta) * np.sin(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(phi)  # Ensuring cameras are above the origin

    poses = []
    for i in range(num_cameras):
        position = np.array([x[i], y[i], z[i]])  # Camera position

        # Compute rotation to look at the origin
        forward = -position / np.linalg.norm(position)  # Camera looks at the origin
        up = np.array([0, 0, 1])  # Define an up vector
        right = np.cross(up, forward)
        right /= np.linalg.norm(right)
        up = np.cross(forward, right)

        # Construct rotation matrix
        R_cam = np.vstack([right, up, forward]).T  # Rotation matrix

        # Construct 4x4 transformation matrix
        T_cam = np.eye(4)
        T_cam[:3, :3] = R_cam
        T_cam[:3, 3] = position

        poses.append(T_cam)

    return poses

# Example usage
num_cameras = 5
poses = generate_camera_poses(num_cameras)

translation = np.array([0.0, -0.55, -0.075]) 
translation_noise_std = 0.015
for i in range(len(poses)):
    noise = np.random.normal(loc=0, scale=translation_noise_std, size=3)
    poses[i][:3, 3] += translation + noise
    
save_camera_poses(poses, f"camera_poses_{num_cameras}_sim.npz")

real_poses = read_real_data("/video_3dqa/yiquan_spad_captures/data/carter_3D_print_poses/A_refined_gt")
save_camera_poses(real_poses, "camera_poses_16_real.npz")

# Visualize the sampled (sim) and real camera positions
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')

# Plot simulated camera positions (red)
for pose in poses:
    ax.scatter(*pose[:3, 3], c='r', marker='o', label="Simulated" if 'simulated' not in locals() else "")
    simulated = True

# Plot real camera positions (blue)
for pose in real_poses:
    ax.scatter(*pose[:3, 3], c='b', marker='^', label="Real" if 'real' not in locals() else "")
    real = True

# Plot a semi-sphere for reference
r = 0.4  # Define the hemisphere radius
u = np.linspace(0, 2 * np.pi, 50)  # Azimuth angle (0 to 360 degrees)
v = np.linspace(0, np.pi / 2, 50)  # Elevation angle (0 to 90 degrees)

# Correcting the scaling of radius
x = np.outer(np.cos(u), np.sin(v)) * r
y = np.outer(np.sin(u), np.sin(v)) * r
z = np.outer(np.ones(np.size(u)), np.cos(v)) * r  # Scale z by r

ax.plot_wireframe(x, y, z, color='gray', alpha=0.3)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# Add legend
ax.legend()

# Save the figure
plt.savefig("camera_poses_comparison.png", dpi=300, bbox_inches='tight')




import numpy as np
import open3d as o3d

def create_sphere(center, radius=0.02, color=[1, 0, 0]):
    """Create a sphere mesh at a given center position."""
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.paint_uniform_color(color)
    sphere.translate(center)
    return sphere

def create_cylinder(pose, length=0.05, radius=0.01, color=[1, 0, 0]):
    """Create a cylinder and apply the camera pose transformation."""
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length)
    cylinder.paint_uniform_color(color)

    # Initial cylinder is along Z-axis in local coordinates
    cylinder.translate([0, 0, length / 2])  # Move to align with the center

    # Apply rotation and translation using the pose matrix
    R_cam = pose[:3, :3]  # Extract rotation matrix
    t_cam = pose[:3, 3]   # Extract translation vector

    cylinder.rotate(R_cam, center=[0, 0, 0])  # Apply camera rotation
    cylinder.translate(t_cam)  # Apply camera translation

    return cylinder

def save_camera_mesh(poses_sim, poses_real, filename="camera_poses.ply", line_length=0.2):
    """
    Save simulated and real camera poses as a single mesh (.ply), using spheres for cameras and cylinders for thick lines.

    Args:
        poses_sim (list of np.ndarray): List of 4x4 transformation matrices for simulated cameras.
        poses_real (list of np.ndarray): List of 4x4 transformation matrices for real cameras.
        filename (str): Output file name.
        line_length (float): Length of the orientation lines.
    """
    mesh_list = []

    # Process Simulated Camera Poses (Red)
    for pose in poses_sim:
        cam_pos = pose[:3, 3]  # Camera position

        mesh_list.append(create_sphere(cam_pos, radius=0.02, color=[1, 0, 0]))  # Red spheres
        mesh_list.append(create_cylinder(pose, length=line_length, radius=0.01, color=[1, 0, 0]))  # Red cylinders

    # Process Real Camera Poses (Blue)
    for pose in poses_real:
        cam_pos = pose[:3, 3]  # Camera position

        mesh_list.append(create_sphere(cam_pos, radius=0.02, color=[0, 0, 1]))  # Blue spheres
        mesh_list.append(create_cylinder(pose, length=line_length, radius=0.01, color=[0, 0, 1]))  # Blue cylinders

    # Combine all meshes into one
    full_mesh = o3d.geometry.TriangleMesh()
    for mesh in mesh_list:
        full_mesh += mesh

    # Save as a single .ply file
    o3d.io.write_triangle_mesh(filename, full_mesh)
    print(f"Saved camera poses mesh with thick lines to {filename}")

# # Example usage:
# num_cameras = 15
# poses_sim = generate_camera_poses(num_cameras)
# poses_real = read_real_data("/video_3dqa/yiquan_spad_captures/data/carter_3D_print_poses/A_refined_gt")

poses_sim = poses
poses_real = real_poses
save_camera_mesh(poses_sim, poses_real, f"camera_poses_{num_cameras}.ply")
