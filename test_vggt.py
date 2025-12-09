import torch
import os
import numpy as np
from PIL import Image
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images


def main():
    device = "cpu"
    print("Running on:", device)

    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()

    kitchen_folder = "examples/kitchen/images"
    image_paths = sorted(
        os.path.join(kitchen_folder, f)
        for f in os.listdir(kitchen_folder)
        if f.endswith(".png")
    )

    print("Loaded", len(image_paths), "kitchen images")

    # use every 3rd image
    # image_paths = image_paths[::3]
    print("Using", len(image_paths), "images:", image_paths)

    images = load_and_preprocess_images(image_paths).to(device)

    with torch.no_grad():
        preds = model(images)

    print("Prediction keys:", preds.keys())
    depth = preds["depth"]
    wp = preds["world_points"]

    print("Depth shape:", depth.shape)
    print("World points shape:", wp.shape)

    os.makedirs("kitchen_depth_out", exist_ok=True)

    # Save depth maps
    depth_cpu = depth[0].cpu()
    V = depth_cpu.shape[0]
    for i in range(V):
        d = depth_cpu[i].squeeze(-1)
        d_np = d.numpy()
        np.save(f"kitchen_depth_out/depth_{i:02d}.npy", d_np)

        d_norm = (d - d.min()) / (d.max() - d.min() + 1e-8)
        d_img = (d_norm * 255).clamp(0, 255).byte().numpy()
        Image.fromarray(d_img, mode="L").save(f"kitchen_depth_out/depth_{i:02d}.png")

    # Save world points (3D coordinates)
    wp_np = wp[0].cpu().numpy()  # [V, H, W, 3]
    np.save("kitchen_depth_out/world_points.npy", wp_np)

    print("Saved world points and depth.")
    print("Done.")


if __name__ == "__main__":
    main()
