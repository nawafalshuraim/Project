# ================================================================
# idss/visualize.py - Results Visualization
# ================================================================
# Creates a 4-panel figure showing the full IDSS decision visually.
# This is what gets saved to the outputs/ folder.
#
# 4 panels:
#   Panel 1 - Original Image
#             The raw grayscale NIR hand photo
#
#   Panel 2 - Vein Mask + Skeleton
#             Blue overlay = detected vein regions
#             Yellow lines = vein centerlines (skeleton)
#
#   Panel 3 - Ranked Segments
#             All accepted segments color coded:
#             Green  = highest safety score (best)
#             Yellow = medium safety score
#             Red    = lowest safety score (worst)
#             Top 3 segments labeled with #1 #2 #3
#
#   Panel 4 - Recommended Insertion Point
#             Best segment highlighted in green
#             Yellow circle = insertion point location
#             Red dot       = exact needle position
#             Arrow + label = INSERT HERE
#             Safety score  = shown in top left corner
# ================================================================

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def visualize_idss(img_np, mask_sm, skeleton, segments,
                   results_df, best_feat, insertion_point,
                   dist_map, final_scores, accepted_indices,
                   save_path=None):
    """
    Create 4-panel IDSS visualization and optionally save it.

    Parameters:
        img_np           : (H, W) float32 grayscale image 0.0-1.0
        mask_sm          : (H, W) uint8 smoothed binary mask
                           can be None if not available
        skeleton         : (H, W) uint8 binary skeleton
        segments         : list of all segment paths
        results_df       : pandas DataFrame with ranked segments
        best_feat        : feature dict of the best segment
        insertion_point  : (y, x) tuple from find_insertion_point()
        dist_map         : (H, W) float32 distance transform
        final_scores     : list of final scores for accepted segments
        accepted_indices : list of original segment indices accepted
        save_path        : file path to save figure (optional)
                           if None figure is only shown not saved
    """
    H, W = img_np.shape

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    fig.suptitle(
        "IDSS - Safest Vein Segment for IV Insertion",
        fontsize=14,
        fontweight="bold",
        y=1.02
    )

    # ================================================================
    # Panel 1 - Original Image
    # ================================================================
    axes[0].imshow(img_np, cmap="gray")
    axes[0].set_title("Original Image", fontweight="bold")
    axes[0].axis("off")

    # ================================================================
    # Panel 2 - Vein Mask + Skeleton
    # ================================================================
    overlay = np.stack([img_np] * 3, axis=-1)

    if mask_sm is not None:
        mask_rgb = np.zeros((H, W, 3))
        mask_rgb[mask_sm > 0] = [0, 0.5, 1.0]
        overlay  = np.clip(overlay + 0.3 * mask_rgb, 0, 1)

    skel_rgb = np.zeros((H, W, 3))
    skel_rgb[skeleton > 0] = [1.0, 1.0, 0.0]
    overlay = np.clip(overlay + 0.6 * skel_rgb, 0, 1)

    axes[1].imshow(overlay)
    axes[1].set_title("Vein Mask + Skeleton", fontweight="bold")
    axes[1].axis("off")

    # ================================================================
    # Panel 3 - Ranked Segments
    # ================================================================
    seg_vis = cv2.cvtColor(
        (np.clip(img_np, 0, 1) * 255).astype(np.uint8),
        cv2.COLOR_GRAY2RGB
    )

    if len(final_scores) > 0:
        score_arr = np.array(final_scores)
        s_min     = score_arr.min()
        s_max     = score_arr.max()

        for rank_i, idx in enumerate(np.argsort(final_scores)[::-1]):
            if idx >= len(accepted_indices):
                continue

            path = segments[accepted_indices[idx]]

            s = (final_scores[idx] - s_min) / (s_max - s_min + 1e-9)

            color = (
                int(255 * (1 - s)),
                int(255 * s),
                0
            )

            for (y, x) in path:
                cv2.circle(seg_vis, (int(x), int(y)), 2, color, -1)

            if rank_i < 3:
                my = int(np.mean([p[0] for p in path]))
                mx = int(np.mean([p[1] for p in path]))
                cv2.putText(
                    seg_vis,
                    f"#{rank_i + 1}",
                    (mx + 3, my),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1, cv2.LINE_AA
                )

    axes[2].imshow(seg_vis)
    axes[2].set_title(
        "Segments Ranked\n(Green=Best, Red=Worst)",
        fontweight="bold"
    )
    axes[2].axis("off")

    # ================================================================
    # Panel 4 - Recommended Insertion Point
    # ================================================================
    rec_vis = cv2.cvtColor(
        (np.clip(img_np, 0, 1) * 255).astype(np.uint8),
        cv2.COLOR_GRAY2RGB
    )

    if best_feat is not None:
        for (y, x) in best_feat["path"]:
            cv2.circle(rec_vis, (int(x), int(y)), 2, (0, 255, 0), -1)

    if insertion_point is not None:
        iy, ix = insertion_point

        cv2.circle(rec_vis, (ix, iy), 12, (255, 255, 0), 2)
        cv2.circle(rec_vis, (ix, iy), 4,  (0, 0, 255), -1)

        cv2.arrowedLine(
            rec_vis,
            (ix + 35, iy - 35),
            (ix + 13, iy - 13),
            (255, 255, 0),
            2,
            tipLength=0.4
        )

        cv2.putText(
            rec_vis,
            "INSERT HERE",
            (ix + 38, iy - 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45, (255, 255, 0), 1, cv2.LINE_AA
        )

    if results_df is not None and len(results_df) > 0:
        best_score  = results_df.iloc[0]["Safety %"]
        score_text  = f"Safety: {best_score}%"
        score_color = (
            (0, 255, 0)      if best_score >= 75
            else (0, 165, 255) if best_score >= 50
            else (0, 0, 255)
        )
        cv2.putText(
            rec_vis,
            score_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, score_color, 2, cv2.LINE_AA
        )

    axes[3].imshow(rec_vis)
    axes[3].set_title(
        "RECOMMENDED INSERTION POINT",
        fontweight="bold",
        color="green"
    )
    axes[3].axis("off")

    # ================================================================
    # Legend
    # ================================================================
    legend = [
        mpatches.Patch(color="green",  label="Best segment"),
        mpatches.Patch(color="yellow", label="Insertion point"),
        mpatches.Patch(color="red",    label="Lower ranked segments"),
    ]
    fig.legend(
        handles=legend,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.05)
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  Visualization saved: {save_path}")

    plt.show()
    plt.close()