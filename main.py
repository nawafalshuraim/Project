import os
import json
import time
import argparse
import torch
import cv2
import numpy as np
from skimage.morphology import skeletonize

# Our modules
from config import (
    MODEL_PATH, MODEL_CHANNELS,
    IMAGE_H, IMAGE_W,
    THRESHOLD,
    MIN_SEGMENT_LEN, SPUR_PRUNE_ITERS
)
from utils.preprocessing import preprocess_image, smooth_mask
from utils.skeleton      import prune_spurs, extract_graph_segments
from idss.pipeline       import run_idss


# Load Model
def load_model():
    """
    Load the TorchScript model from config.py MODEL_PATH.
    When you update your model, just change MODEL_PATH in config.py.
    Nothing else needs to change.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            f"Make sure vein_model.pt is in the project folder."
        )

    model = torch.jit.load(MODEL_PATH, map_location="cpu")
    model.eval()
    print(f"Model loaded: {MODEL_PATH}")
    print(f"Channels: {MODEL_CHANNELS}")
    return model


# Process One Image
def run_on_image(model, img_path, save_path=None):
    """
    Full pipeline for one hand image:
      1. Load and preprocess image
      2. Run model -> get probability map
      3. Clean mask -> distance transform
      4. Skeletonize -> extract segments
      5. Run IDSS -> rank segments -> find insertion point
      6. Visualize results
      7. Save JSON output for UI designer
    """
    print(f"\n{'─'*50}")
    print(f"Image: {img_path}")
    print(f"{'─'*50}")

    # 1. Load and preprocess
    img_u8, tensor = preprocess_image(img_path, IMAGE_H, IMAGE_W)
    img_np = img_u8.astype(np.float32) / 255.0

    # 2. Run model
    with torch.no_grad():
        out     = model(tensor)                        # (1, C, H, W)
        prob_np = torch.sigmoid(out[0, 0]).numpy()     # (H, W)

    print(f"  Prob range:     {prob_np.min():.3f} -> {prob_np.max():.3f}")
    print(f"  Vein pixels:    {(prob_np > THRESHOLD).sum()}")
    print(f"  Vein coverage:  {100*(prob_np > THRESHOLD).mean():.1f}%")

    # 3. Clean mask + distance transform
    mask_raw = (prob_np > THRESHOLD).astype(np.uint8)
    mask_sm  = smooth_mask(mask_raw)
    dist_map = cv2.distanceTransform(mask_sm, cv2.DIST_L2, 5)

    # 4. Skeleton
    skeleton = skeletonize(mask_sm > 0).astype(np.uint8)
    skeleton = prune_spurs(skeleton, iters=SPUR_PRUNE_ITERS)

    # 5. Segments
    segments = extract_graph_segments(skeleton, min_len_px=MIN_SEGMENT_LEN)
    print(f"  Segments found: {len(segments)}")

    if len(segments) == 0:
        print("  No segments found - try a different image.")
        return None, None, None

    # 6. Run IDSS
    results_df, best_feat, insertion_point = run_idss(
        img_np       = img_np,
        prob_np      = prob_np,
        skeleton     = skeleton,
        dist_map     = dist_map,
        segments     = segments,
        num_to_show  = 5,
        verbose      = True,
        save_path    = save_path,
    )

    # 7. Save JSON output for UI designer
    if results_df is not None and best_feat is not None:
        output_json = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "image":     os.path.basename(img_path),
            "status":    "success",

            "best_segment": {
                "segment_id":   int(results_df.iloc[0]["Segment"]),
                "safety_score": float(results_df.iloc[0]["Safety %"]),
                "rank":         1
            },

            "insertion_point": {
                "x":          int(insertion_point[1]),
                "y":          int(insertion_point[0]),
                "confidence": float(round(best_feat["confidence"] * 100, 1))
            },

            "vein_metrics": {
                "length_px":   float(round(best_feat["length_px"], 1)),
                "diameter_px": float(round(best_feat["diameter_px"], 2)),
                "tortuosity":  float(round(best_feat["tortuosity"], 3)),
                "region":      str(results_df.iloc[0]["Region"])
            },

            "all_segments": [
                {
                    "rank":           int(row["Rank"]),
                    "segment_id":     int(row["Segment"]),
                    "safety_percent": float(row["Safety %"]),
                    "length_px":      float(row["Length (px)"]),
                    "diameter_px":    float(row["Diameter (px)"]),
                    "confidence":     float(row["Confidence"]),
                    "tortuosity":     float(row["Tortuosity"]),
                    "region":         str(row["Region"]),
                }
                for _, row in results_df.iterrows()
            ],

            "summary": {
                "total_segments":    len(segments),
                "accepted_segments": len(results_df),
                "rejected_segments": len(segments) - len(results_df),
            }
        }

        # Save JSON next to image output
        json_path = (
            save_path.replace(".png", ".json")
            if save_path else
            f"outputs/{os.path.basename(img_path)}.json"
        )
        os.makedirs("outputs", exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(output_json, f, indent=2)
        print(f"\n  JSON saved: {json_path}")
        print(json.dumps(output_json, indent=2))

    return results_df, best_feat, insertion_point


# Process a Folder of Images
def run_on_folder(model, folder_path, output_dir="outputs"):
    """
    Run IDSS on every image in a folder.
    Results saved to outputs/ folder automatically.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Find all image files
    exts  = {".png", ".jpg", ".jpeg", ".bmp"}
    files = [
        f for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in exts
    ]

    if not files:
        print(f"No images found in {folder_path}")
        return

    print(f"\nFound {len(files)} images in {folder_path}")

    for fname in sorted(files):
        img_path  = os.path.join(folder_path, fname)
        save_path = os.path.join(output_dir, f"idss_{fname}")
        try:
            run_on_image(model, img_path, save_path=save_path)
        except Exception as e:
            print(f"  Error on {fname}: {e}")


# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="IDSS - Safest Vein Segment Finder for IV Insertion"
    )
    parser.add_argument("--image",  type=str, help="Path to a single hand image")
    parser.add_argument("--folder", type=str, help="Path to folder of hand images")
    parser.add_argument("--save",   type=str, help="Path to save output image")
    args = parser.parse_args()

    # Load model once
    model = load_model()

    if args.image:
        # Run on single image
        run_on_image(model, args.image, save_path=args.save)

    elif args.folder:
        # Run on all images in folder
        run_on_folder(model, args.folder)

    else:
        # No image provided - just test model loads correctly
        print("\nNo image provided - testing model only...")
        dummy = torch.zeros(1, 1, IMAGE_H, IMAGE_W)
        with torch.no_grad():
            out = model(dummy)
        print(f"  Output shape: {out.shape}")
        print(f"\nModel is ready.")
        print(f"Usage:")
        print(f"  python main.py --image images/hand001.png")
        print(f"  python main.py --folder images/")