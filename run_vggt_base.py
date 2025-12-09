import os
import torch
import numpy as np
from PIL import Image

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images


def main():
    # pick device – for now, cpu is safest on mac
    device = "cpu"
    print("running on:", device)

    # load smaller vggt-base from huggingface
    # this uses the same from_pretrained mechanism you used for vggt-1b
    model = VGGT.from_pretrained("facebook/VGGT-Base").to(device)
    model.eval()

    # example: use kitchen images first to verify everything works
    kitchen_folder = "examples/kitchen/images"
    image_paths = sorted(
        os.path.join(kitchen_folder, f)
        for f in os.listdir(kitchen_folder)
        if f.endswith(".png")
    )

    print("loaded", len(image_paths), "images")

    # you can subsample if you want fewer views
    # image_paths = image_paths[::3]

    # preprocess images into tensor [B, S, 3, H, W]
    images = load_and_preprocess_images(image_paths).to(device)

    with torch.no_grad():
        preds = model(images)

    print("prediction keys:", preds.keys())

    # depth: [B, S, H, W, 1]
    depth = preds["depth"]
    print("depth shape:", depth.shape)

    # optional: world_points (these are in vggt’s learned space)
    if "world_points" in preds:
        print("world_points shape:", preds["world_points"].shape)

    # save depths so you can visually inspect them
    out_dir = "vggt_base_depth_out"
    os.makedirs(out_dir, exist_ok=True)

    depth_cpu = depth[0].cpu()  # [S, H, W, 1]
    num_views = depth_cpu.shape[0]

    for i in range(num_views):
        d = depth_cpu[i].squeeze(-1)  # [H, W]
        d_np = d.numpy()
        np.save(os.path.join(out_dir, f"depth_{i:02d}.npy"), d_np)

        # normalize for visualization
        d_norm = (d - d.min()) / (d.max() - d.min() + 1e-8)
        d_img = (d_norm * 255).clamp(0, 255).byte().numpy()
        Image.fromarray(d_img, mode="L").save(
            os.path.join(out_dir, f"depth_{i:02d}.png")
        )

        print("saved view", i, "to", os.path.join(out_dir, f"depth_{i:02d}.png"))

    print("done.")


if __name__ == "__main__":
    main()
