# ================================================================
# idss/normalize.py - Feature Normalization
# ================================================================
# Scales all features to the same 0.0 to 1.0 range.
#
# Why normalization is critical:
#   Without it, features with large values dominate the score.
#
#   Example without normalization:
#     length    = 257 px   <- huge number, dominates everything
#     diameter  = 15 px
#     confidence = 0.99    <- tiny number, ignored
#
#   Example after normalization:
#     length_norm    = 1.00  <- all on same scale
#     diameter_norm  = 0.75  <- fair comparison
#     confidence_norm = 0.95 <- contributes properly
#
# Method used: Min-Max normalization
#   x_norm = (x - min) / (max - min)
#
#   Result:
#     Worst segment gets 0.0 for that feature
#     Best segment gets  1.0 for that feature
#     All others fall between 0.0 and 1.0
#
# Special case - Tortuosity is INVERTED:
#   Lower tortuosity = straighter vein = better
#   So we flip it: tortuosity_norm = 1.0 - minmax(tortuosity)
#   Now 0.0 = most curved (worst) and 1.0 = straightest (best)
# ================================================================


def normalize_features(feature_list):
    """
    Min-max normalize all features across all accepted segments.

    Normalization is done ACROSS segments not per segment.
    This means each feature is scaled relative to all other
    segments in the same image.

    Parameters:
        feature_list : list of feature dicts from extract_idss_features()
                       only accepted segments should be passed in

    Returns:
        List of normalized feature dicts
        Each dict has keys ending in _norm
        All values are between 0.0 and 1.0
    """
    if len(feature_list) == 0:
        return []

    # The 7 features used in TOPSIS scoring
    # These match the 7 criteria in the AHP matrix
    keys = [
        "length_px",        # Feature 1
        "diameter_px",      # Feature 2
        "confidence",       # Feature 3
        "tortuosity",       # Feature 4 (will be inverted)
        "branch_distance",  # Feature 5
        "edge_distance",    # Feature 6
        "region_score",     # Feature 7
    ]

    # ================================================================
    # Step 1 - Find min and max for each feature
    # ================================================================
    # We need these to scale every value to 0.0 - 1.0
    ranges = {}
    for k in keys:
        vals      = [f[k] for f in feature_list]
        ranges[k] = (min(vals), max(vals))

    # ================================================================
    # Step 2 - Min-max normalization helper
    # ================================================================
    def minmax(val, lo, hi):
        """
        Scale a single value to 0.0 - 1.0 range.

        Special case: if all segments have the same value
        (hi - lo = 0), return 1.0 for all.
        This avoids division by zero and treats all segments equally.

        Example:
            val=150, lo=50, hi=250  ->  (150-50)/(250-50) = 0.50
            val=50,  lo=50, hi=250  ->  (50-50)/(250-50)  = 0.00
            val=250, lo=50, hi=250  ->  (250-50)/(250-50) = 1.00
        """
        if hi - lo < 1e-9:
            return 1.0
        return (val - lo) / (hi - lo)

    # ================================================================
    # Step 3 - Normalize each segment
    # ================================================================
    normalized = []

    for feat in feature_list:
        normalized.append({

            # Length: higher is better
            # Longer vein = more room for insertion
            "length_norm": minmax(
                feat["length_px"], *ranges["length_px"]
            ),

            # Diameter: higher is better
            # Wider vein = easier catheter insertion
            "diameter_norm": minmax(
                feat["diameter_px"], *ranges["diameter_px"]
            ),

            # Confidence: higher is better
            # More certain detection = more reliable
            "confidence_norm": minmax(
                feat["confidence"], *ranges["confidence"]
            ),

            # Tortuosity: INVERTED - lower is better
            # Straighter vein = easier needle insertion
            # We flip so that 1.0 = straightest (best)
            "tortuosity_norm": 1.0 - minmax(
                feat["tortuosity"], *ranges["tortuosity"]
            ),

            # Branch distance: higher is better
            # Further from branches = safer insertion zone
            "branch_dist_norm": minmax(
                feat["branch_distance"], *ranges["branch_distance"]
            ),

            # Edge distance: higher is better
            # Further from border = more stable position
            "edge_dist_norm": minmax(
                feat["edge_distance"], *ranges["edge_distance"]
            ),

            # Region score: higher is better
            # Distal (1.0) > Middle (0.6) > Proximal (0.1)
            "region_norm": minmax(
                feat["region_score"], *ranges["region_score"]
            ),
        })

    return normalized