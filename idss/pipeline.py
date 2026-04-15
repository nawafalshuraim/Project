# ================================================================
# idss/pipeline.py - Main IDSS Orchestrator
# ================================================================
# This is the brain of the IDSS.
# It connects all other files in the correct order
# and produces the final ranked results.
#
# Order of operations:
#   1. Feature Extraction  (features.py)
#      -> measure every segment
#
#   2. Knowledge Rules     (rules.py)
#      -> accept or reject each segment
#      -> apply penalties and bonuses
#
#   3. AHP Weights         (ahp.py)
#      -> determine importance of each feature
#
#   4. Normalization       (normalize.py)
#      -> scale all features to 0.0-1.0
#
#   5. TOPSIS Scoring      (topsis.py)
#      -> rank segments by similarity to ideal vein
#
#   6. Final Scoring
#      -> apply penalties and bonuses from rules
#      -> produce final safety percentage
#
#   7. Insertion Point     (insertion.py)
#      -> find exact needle position on best segment
#
#   8. Visualization       (visualize.py)
#      -> draw results on image
#      -> save to outputs folder
# ================================================================

import numpy as np
import pandas as pd

from idss.features  import extract_idss_features
from idss.rules     import apply_knowledge_rules
from idss.ahp       import compute_ahp_weights
from idss.normalize import normalize_features
from idss.topsis    import topsis_score
from idss.insertion import find_insertion_point
from idss.visualize import visualize_idss


def run_idss(img_np, prob_np, skeleton, dist_map, segments,
             num_to_show=5, verbose=True, save_path=None):
    """
    Run the complete IDSS pipeline for one image.

    Parameters:
        img_np      : (H, W) float32 grayscale image 0.0-1.0
        prob_np     : (H, W) float32 model probability map
        skeleton    : (H, W) uint8 binary skeleton
        dist_map    : (H, W) float32 distance transform
        segments    : list of segment paths from extract_graph_segments()
        num_to_show : how many top segments to print in table
        verbose     : print detailed output to console
        save_path   : file path to save visualization image

    Returns:
        results_df      : pandas DataFrame - all ranked segments
        best_feat       : dict - feature dict of best segment
        insertion_point : (y, x) tuple - recommended needle position
    """
    H, W = img_np.shape

    # Handle empty segments
    if len(segments) == 0:
        print("No segments found for this image.")
        return None, None, None

    # ================================================================
    # Step 1 - Feature Extraction
    # ================================================================
    features = extract_idss_features(
        segments  = segments,
        skeleton  = skeleton,
        dist_map  = dist_map,
        prob_np   = prob_np,
        img_shape = (H, W)
    )

    # ================================================================
    # Step 2 - Knowledge Rules
    # ================================================================
    accepted_features = []
    accepted_indices  = []
    rule_results      = []

    for i, feat in enumerate(features):
        acc, penalty, bonus, reasons = apply_knowledge_rules(feat)

        rule_results.append({
            "accepted": acc,
            "penalty":  penalty,
            "bonus":    bonus,
            "reasons":  reasons,
        })

        if acc:
            accepted_features.append(feat)
            accepted_indices.append(i)

    if verbose:
        print(f"\n{'='*55}")
        print(f"IDSS Analysis")
        print(f"{'='*55}")
        print(f"Total segments:    {len(segments)}")
        print(f"Accepted:          {len(accepted_features)}")
        print(f"Rejected:          {len(segments) - len(accepted_features)}")

    if len(accepted_features) == 0:
        print("All segments rejected by knowledge rules.")
        print("Try adjusting thresholds in config.py")
        return None, None, None

    # ================================================================
    # Step 3 - AHP Weights
    # ================================================================
    weights, CR = compute_ahp_weights(verbose=verbose)

    # ================================================================
    # Step 4 - Normalize Features
    # ================================================================
    normalized = normalize_features(accepted_features)

    # ================================================================
    # Step 5 - TOPSIS Scoring
    # ================================================================
    t_scores = topsis_score(normalized, weights)

    # ================================================================
    # Step 6 - Apply Penalties and Bonuses
    # ================================================================
    final_scores = []
    for i, score in enumerate(t_scores):
        rr    = rule_results[accepted_indices[i]]
        final = min(1.0, score * rr["penalty"] * rr["bonus"])
        final_scores.append(final)

    # ================================================================
    # Step 7 - Rank Segments
    # ================================================================
    ranked = np.argsort(final_scores)[::-1]

    rows = []
    for rank, idx in enumerate(ranked):
        feat = accepted_features[idx]
        rr   = rule_results[accepted_indices[idx]]

        if feat["region_score"] >= 1.0:
            region = "Distal"
        elif feat["region_score"] >= 0.6:
            region = "Middle"
        else:
            region = "Proximal"

        rows.append({
            "Rank":          rank + 1,
            "Segment":       accepted_indices[idx] + 1,
            "Score":         round(final_scores[idx], 4),
            "Safety %":      round(final_scores[idx] * 100, 1),
            "Length (px)":   round(feat["length_px"], 1),
            "Diameter (px)": round(feat["diameter_px"], 2),
            "Confidence":    round(feat["confidence"], 3),
            "Tortuosity":    round(feat["tortuosity"], 3),
            "Branch Dist":   round(feat["branch_distance"], 1),
            "Edge Dist":     round(feat["edge_distance"], 1),
            "Region":        region,
            "Notes":         " | ".join(rr["reasons"]) if rr["reasons"] else "-",
        })

    results_df = pd.DataFrame(rows)

    # ================================================================
    # Step 8 - Best Segment and Insertion Point
    # ================================================================
    best_idx        = ranked[0]
    best_feat       = accepted_features[best_idx]
    insertion_point = find_insertion_point(
        path     = best_feat["path"],
        dist_map = dist_map
    )

    # ================================================================
    # Print Results
    # ================================================================
    if verbose:
        print(f"\n{'='*55}")
        print(f"TOP {min(num_to_show, len(rows))} SEGMENTS")
        print(f"{'='*55}")
        print(
            results_df[[
                "Rank", "Segment", "Safety %",
                "Length (px)", "Diameter (px)",
                "Confidence", "Tortuosity", "Region"
            ]].head(num_to_show).to_string(index=False)
        )

        print(f"\n{'='*55}")
        print(f"BEST SEGMENT: Segment {accepted_indices[best_idx] + 1}")
        print(f"  Safety Score:    {final_scores[best_idx]*100:.1f}%")
        print(f"  Length:          {best_feat['length_px']:.1f} px")
        print(f"  Diameter:        {best_feat['diameter_px']:.2f} px")
        print(f"  Confidence:      {best_feat['confidence']:.3f}")
        print(f"  Conf Variation:  {best_feat['conf_variation']*100:.1f}%")
        print(f"  Tortuosity:      {best_feat['tortuosity']:.3f}")
        print(f"  Branch Dist:     {best_feat['branch_distance']:.1f} px")
        print(f"  Insertion Point: y={insertion_point[0]}, x={insertion_point[1]}")
        print(f"{'='*55}")

        print(f"\nFull Ranked Table:")
        print(
            results_df[[
                "Rank", "Segment", "Safety %",
                "Length (px)", "Diameter (px)",
                "Confidence", "Tortuosity",
                "Region", "Notes"
            ]].to_string(index=False)
        )

    # ================================================================
    # Step 9 - Visualization
    # ================================================================
    visualize_idss(
        img_np           = img_np,
        mask_sm          = None,
        skeleton         = skeleton,
        segments         = segments,
        results_df       = results_df,
        best_feat        = best_feat,
        insertion_point  = insertion_point,
        dist_map         = dist_map,
        final_scores     = final_scores,
        accepted_indices = accepted_indices,
        save_path        = save_path,
    )

    return results_df, best_feat, insertion_point