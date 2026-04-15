# ================================================================
# idss/topsis.py - TOPSIS Multi-Criteria Scoring
# ================================================================
# Ranks all accepted vein segments by how close each one is
# to the ideal perfect vein for IV insertion.
#
# TOPSIS = Technique for Order Preference by
#          Similarity to Ideal Solution
#
# The core idea:
#   Imagine the perfect vein:
#     - maximum length
#     - maximum diameter
#     - maximum confidence
#     - minimum tortuosity
#     - maximum branch distance
#     - maximum edge distance
#     - maximum region score
#
#   And the worst possible vein (opposite of all above).
#
#   TOPSIS measures:
#     - How far each segment is from the worst vein
#     - How far each segment is from the best vein
#
#   Final score:
#     score = dist_to_worst / (dist_to_best + dist_to_worst)
#
#   Score range:
#     1.0 = identical to ideal vein    (best possible)
#     0.0 = identical to worst vein    (worst possible)
#
# Why TOPSIS over simple weighted sum:
#   Weighted sum: score = w1*f1 + w2*f2 + ...
#     Problem: a very long vein compensates for a thin vein
#
#   TOPSIS: considers ALL criteria simultaneously
#     A segment must be close to ideal on ALL criteria
#     One very good feature cannot hide a bad feature
# ================================================================

import numpy as np


def topsis_score(normalized_list, weights):
    """
    Rank segments using TOPSIS multi-criteria decision making.

    Parameters:
        normalized_list : list of normalized feature dicts
                          from normalize_features()
                          all values already 0.0 to 1.0

        weights         : (7,) numpy array from compute_ahp_weights()
                          importance weight per criterion

    Returns:
        List of float scores, one per segment
        Higher score = better segment for IV insertion
        Range: 0.0 (worst) to 1.0 (best)
    """
    if len(normalized_list) == 0:
        return []

    # The 7 normalized features in the same order as AHP weights
    keys = [
        "length_norm",       # Weight index 0
        "diameter_norm",     # Weight index 1
        "confidence_norm",   # Weight index 2
        "tortuosity_norm",   # Weight index 3
        "branch_dist_norm",  # Weight index 4
        "edge_dist_norm",    # Weight index 5
        "region_norm",       # Weight index 6
    ]

    # ================================================================
    # Step 1 - Build Decision Matrix
    # ================================================================
    # Rows    = segments (one row per segment)
    # Columns = criteria (one column per feature)
    #
    # Example with 3 segments and 7 criteria:
    #   [[1.00, 1.00, 0.50, 0.90, 0.80, 0.70, 1.00],  <- segment 1
    #    [0.00, 0.33, 0.00, 0.80, 0.60, 1.00, 0.60],  <- segment 2
    #    [0.41, 0.00, 1.00, 1.00, 1.00, 0.50, 0.60]]  <- segment 3
    matrix = np.array(
        [[n[k] for k in keys] for n in normalized_list]
    )

    # ================================================================
    # Step 2 - Apply AHP Weights
    # ================================================================
    # Multiply each column by its importance weight
    # Diameter column gets highest weight (0.30)
    # Tortuosity column gets lowest weight (0.04)
    #
    # This ensures diameter matters 7x more than tortuosity
    # in the final ranking
    weighted = matrix * weights

    # ================================================================
    # Step 3 - Find Ideal and Anti-Ideal Solutions
    # ================================================================
    # Ideal solution      = best possible value in each column
    #                       (max of each weighted criterion)
    # Anti-ideal solution = worst possible value in each column
    #                       (min of each weighted criterion)
    #
    # These are theoretical reference points, not real segments
    ideal      = weighted.max(axis=0)  # best  in each criterion
    anti_ideal = weighted.min(axis=0)  # worst in each criterion

    # ================================================================
    # Step 4 - Calculate Distances
    # ================================================================
    # For each segment, calculate:
    #   dist_to_ideal      = Euclidean distance to ideal solution
    #   dist_to_anti_ideal = Euclidean distance to anti-ideal solution
    #
    # Euclidean distance in 7-dimensional space:
    #   dist = sqrt( (w1-i1)^2 + (w2-i2)^2 + ... + (w7-i7)^2 )
    #
    # A segment close to ideal     -> small dist_to_ideal
    # A segment far from anti-ideal -> large dist_to_anti_ideal
    dist_to_ideal      = np.sqrt(
        ((weighted - ideal) ** 2).sum(axis=1)
    )
    dist_to_anti_ideal = np.sqrt(
        ((weighted - anti_ideal) ** 2).sum(axis=1)
    )

    # ================================================================
    # Step 5 - Compute TOPSIS Score
    # ================================================================
    # Score = how much of the total distance is away from anti-ideal
    #
    # Formula:
    #   score = dist_to_anti_ideal / (dist_to_ideal + dist_to_anti_ideal)
    #
    # Intuition:
    #   If segment is very close to ideal and far from anti-ideal:
    #     dist_to_ideal      = 0.1  (small)
    #     dist_to_anti_ideal = 0.9  (large)
    #     score = 0.9 / (0.1 + 0.9) = 0.90  (high score)
    #
    #   If segment is far from ideal and close to anti-ideal:
    #     dist_to_ideal      = 0.9  (large)
    #     dist_to_anti_ideal = 0.1  (small)
    #     score = 0.1 / (0.9 + 0.1) = 0.10  (low score)
    #
    # 1e-9 added to denominator to prevent division by zero
    scores = dist_to_anti_ideal / (
        dist_to_ideal + dist_to_anti_ideal + 1e-9
    )

    return scores.tolist()