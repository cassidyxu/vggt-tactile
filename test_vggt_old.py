import torch
import os
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images


def main():
    device = "cpu"  # <--- force CPU
    print("Running on:", device)

    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()

    kitchen_folder = "examples/kitchen/images"
    image_paths = sorted(
        [
            os.path.join(kitchen_folder, f)
            for f in os.listdir(kitchen_folder)
            if f.endswith(".png")
        ]
    )

    # Optional: just use first 4 views to reduce memory
    image_paths = image_paths[:4]
    print("Loaded", len(image_paths), "kitchen images")

    images = load_and_preprocess_images(image_paths).to(device)

    with torch.no_grad():
        preds = model(images)

    print("Prediction keys:", preds.keys())
    depth = preds["depth"]
    print("Depth shape:", depth.shape)

    os.makedirs("kitchen_depth_out", exist_ok=True)

    depth_cpu = depth[0].detach().cpu() if depth.dim() == 4 else depth.detach().cpu()

    from PIL import Image

    for i in range(depth_cpu.shape[1]):  # depth shape is [1, V, H, W, 1]
        d = depth_cpu[0, i].squeeze(-1)  # [H, W]

        d_norm = (d - d.min()) / (d.max() - d.min() + 1e-8)
        d_img = (d_norm * 255).clamp(0, 255).byte().numpy()

        save_path = f"kitchen_depth_out/depth_{i:02d}.png"
        Image.fromarray(d_img, mode="L").save(save_path)
        print("Saved", save_path)


if __name__ == "__main__":
    main()
