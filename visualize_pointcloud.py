import numpy as np
import open3d as o3d
from PIL import Image
import torch
import os
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images


def depth_to_pointcloud(depth, intrinsics):
    """
    depth: H x W numpy array (float)
    intrinsics: 3x3 numpy array
    Returns Nx3 point cloud in camera coordinates
    """
    H, W = depth.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    xs, ys = np.meshgrid(np.arange(W), np.arange(H))

    Z = depth
    X = (xs - cx) * Z / fx
    Y = (ys - cy) * Z / fy

    points = np.stack([X, Y, Z], axis=-1)
    points = points.reshape(-1, 3)
    return points


def main():
    device = "cpu"

    # Load VGGT model
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()

    # Load kitchen images (same as in your test script)
    kitchen_folder = "examples/kitchen/images"
    image_paths = sorted(
        [
            os.path.join(kitchen_folder, f)
            for f in os.listdir(kitchen_folder)
            if f.endswith(".png")
        ]
    )
    image_paths = image_paths[:4]  # use first 4 views

    images = load_and_preprocess_images(image_paths).to(device)

    # Run VGGT
    with torch.no_grad():
        preds = model(images)

    depth = preds["depth"][0]  # shape: [V, H, W, 1]
    cameras = preds["world_points_conf"]  # actually cameras stored differently
    intrinsics_batch = preds["images"]["intrinsics"]  # intrinsics for each view
    extrinsics_batch = preds["images"]["extrinsics"]  # camera extrinsics per view

    # Choose view 0
    d = depth[0].squeeze(-1).cpu().numpy()  # [H, W]
    intr = intrinsics_batch[0].cpu().numpy()  # 3x3
    extr = extrinsics_batch[0].cpu().numpy()  # 4x4

    # Convert depth → camera frame point cloud
    pts_cam = depth_to_pointcloud(d, intr)

    # Convert to world coordinates using extrinsics
    R = extr[:3, :3]
    t = extr[:3, 3]
    pts_world = pts_cam @ R.T + t

    # Filter out invalid depths (too close or too far)
    mask = (
        np.isfinite(pts_world[:, 2]) & (pts_world[:, 2] > 0.05) & (pts_world[:, 2] < 10)
    )
    pts_world = pts_world[mask]

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_world)

    # Visualize
    print("Opening point cloud viewer…")
    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    main()
