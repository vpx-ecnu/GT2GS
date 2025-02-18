import torch
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    BlendParams,
)
from pytorch3d.structures import Meshes
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer.cameras import look_at_view_transform
from pytorch3d.renderer.mesh import TexturesVertex
import matplotlib.pyplot as plt
import numpy as np

# Helper to save depth and image
def save_image_and_depth(image, depth, filename_prefix):
    plt.imsave(f"{filename_prefix}_image.png", image.cpu().numpy())
    np.save(f"{filename_prefix}_depth.npy", depth.cpu().numpy())

# 1. Create a cube mesh at origin
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vertices = torch.tensor(
    [
        [-1, -1, -1],
        [-1, -1,  1],
        [-1,  1, -1],
        [-1,  1,  1],
        [ 1, -1, -1],
        [ 1, -1,  1],
        [ 1,  1, -1],
        [ 1,  1,  1],
    ],
    dtype=torch.float32,
    device=device,
)
faces = torch.tensor(
    [
        [0, 1, 2], [1, 3, 2],
        [4, 5, 6], [5, 7, 6],
        [0, 1, 4], [1, 5, 4],
        [2, 3, 6], [3, 7, 6],
        [0, 2, 4], [2, 6, 4],
        [1, 3, 5], [3, 7, 5],
    ],
    dtype=torch.int64,
    device=device,
)
textures = TexturesVertex(verts_features=torch.ones_like(vertices).unsqueeze(0))
mesh = Meshes(verts=[vertices], faces=[faces], textures=textures)

# 2. Define two cameras
R1, T1 = look_at_view_transform(2.7, 0, 0)  # Camera 1
R2, T2 = look_at_view_transform(2.7, 90, 0)  # Camera 2

cameras1 = PerspectiveCameras(device=device, R=R1, T=T1)
cameras2 = PerspectiveCameras(device=device, R=R2, T=T2)

# 3. Renderer setup
raster_settings = RasterizationSettings(
    image_size=256,
    blur_radius=0.0,
    faces_per_pixel=1,
)
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=cameras1, raster_settings=raster_settings),
    shader=SoftPhongShader(device=device, cameras=cameras1),
)

# 4. Render images and depth
images1, depths1 = renderer(mesh)
save_image_and_depth(images1[0, ..., :3], depths1[0, ..., 0], "camera1")

# Change to the second camera
renderer.rasterizer.cameras = cameras2
images2, depths2 = renderer(mesh)
save_image_and_depth(images2[0, ..., :3], depths2[0, ..., 0], "camera2")

# 5. Depth to 3D points (inverse projection)
def depth_to_3d(depth, cameras):
    fx, fy, cx, cy = cameras.get_focal_length(), cameras.get_principal_point()
    h, w = depth.shape
    y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device))
    z = depth.flatten()
    x = ((x.flatten() - cx) / fx) * z
    y = ((y.flatten() - cy) / fy) * z
    points = torch.stack([x, y, z], dim=1)
    return points

points_3d = depth_to_3d(depths1[0, ..., 0], cameras1)

# Transform points to world space and project with camera2
points_world = torch.matmul(points_3d, R1.transpose(0, 1)) + T1
points_camera2 = torch.matmul(points_world - T2, R2)

# 6. Render the transformed points
# Use points_camera2 to generate a new image using cameras2
# Save the new rendering
