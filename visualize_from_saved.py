import numpy as np
import open3d as o3d

wp = np.load("kitchen_depth_out/world_points.npy")  # [V, H, W, 3]
pts = wp.reshape(-1, 3)

mask = np.isfinite(pts).all(axis=1)
pts = pts[mask]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts)
o3d.visualization.draw_geometries([pcd])
