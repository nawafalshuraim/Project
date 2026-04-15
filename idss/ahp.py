# ================================================================
# Pairwise comparison matrix
# (medical expert judgment encoded as numbers)
#         |
#         v
# Normalize each column
# (make comparable across criteria)
#         |
#         v
# Average each row
# (extract importance weight)
#         |
#         v
# Consistency check (CR < 0.10)
# (verify judgments are not contradictory)
#         |
#         v
# weights array [0.13, 0.30, 0.18, 0.04, 0.10, 0.06, 0.18]
#         |
#         v
# Used by TOPSIS to weight the decision matrix
# ================================================================
# Determines how important each feature is relative to others.
# Based on medical expert priorities for IV insertion.
#
# Instead of guessing weights like:
#   score = 0.5 * length + 2.0 * diameter  (arbitrary)
#
# AHP uses pairwise comparisons:
#   "Is diameter more important than length?"
#   "How much more important? 3 times? 5 times?"
#
# This produces scientifically justified weights that reflect
# real medical priorities.
#
# The 7 criteria being weighted:
#   1. Length        - how long the vein is
#   2. Diameter      - how wide the vein is      <- most important
#   3. Confidence    - model certainty
#   4. Tortuosity    - how curved the vein is
#   5. Branch dist   - distance from branching points
#   6. Edge dist     - distance from image border
#   7. Region        - anatomical location score
# ================================================================

import numpy as np


def compute_ahp_weights(verbose=True):
    """
    Compute feature importance weights using AHP.

    How AHP works:
      1. Build a pairwise comparison matrix
         Each cell (i,j) answers:
         "How much more important is criterion i vs criterion j?"

         Scale:
           1   = equally important
           3   = moderately more important
           5   = strongly more important
           7   = very strongly more important
           9   = extremely more important
           1/3 = moderately less important (reciprocal)
           1/5 = strongly less important   (reciprocal)

      2. Normalize the matrix
         Divide each value by its column sum
         Now all columns sum to 1.0

      3. Compute weights
         Average each row
         This gives the importance weight for each criterion

      4. Check consistency
         Consistency Ratio (CR) must be < 0.10
         CR >= 0.10 means comparisons are contradictory
         and the matrix needs to be revised

    Parameters:
        verbose : print weights and CR to console (default True)

    Returns:
        weights : (7,) numpy array - importance weight per criterion
        CR      : float - consistency ratio (should be < 0.10)
    """

    # ================================================================
    # Pairwise Comparison Matrix
    # ================================================================
    # Rows and columns represent the 7 criteria in this order:
    #   0: Length
    #   1: Diameter
    #   2: Confidence
    #   3: Tortuosity
    #   4: Branch distance
    #   5: Edge distance
    #   6: Region score
    #
    # Reading the matrix:
    #   M[1][0] = 3 means "Diameter is 3x more important than Length"
    #   M[0][1] = 1/3 means "Length is 1/3 as important as Diameter"
    #
    # Medical reasoning behind key comparisons:
    #   Diameter > Length     (3x) - wide vein more critical than long
    #   Diameter > Confidence (2x) - physical size matters most
    #   Diameter > Tortuosity (5x) - diameter is much more critical
    #   Confidence > Length   (2x) - uncertain detection is risky
    #   Region > Edge dist    (3x) - location matters more than border
    #
    #              Len   Diam   Conf   Tort   Branch  Edge   Region
    M = np.array([
        [1,        1/3,   1/2,   3,     2,      3,     1/2],  # Length
        [3,        1,     2,     5,     3,      4,     2  ],  # Diameter
        [2,        1/2,   1,     4,     2,      3,     1  ],  # Confidence
        [1/3,      1/5,   1/4,   1,     1/3,    1/2,   1/4],  # Tortuosity
        [1/2,      1/3,   1/2,   3,     1,      2,     1/2],  # Branch dist
        [1/3,      1/4,   1/3,   2,     1/2,    1,     1/3],  # Edge dist
        [2,        1/2,   1,     4,     2,      3,     1  ],  # Region
    ], dtype=np.float64)

    # ================================================================
    # Step 1 - Normalize columns
    # ================================================================
    # Each value divided by its column sum
    # After this step, each column sums to exactly 1.0
    col_sums = M.sum(axis=0)
    M_norm   = M / col_sums

    # ================================================================
    # Step 2 - Compute weights
    # ================================================================
    # Average each row of the normalized matrix
    # This is the priority weight for each criterion
    weights = M_norm.mean(axis=1)
    weights = weights / weights.sum()  # ensure exactly sums to 1.0

    # ================================================================
    # Step 3 - Consistency Check
    # ================================================================
    # AHP requires that comparisons are logically consistent.
    # Example of inconsistency:
    #   A is 3x more important than B
    #   B is 3x more important than C
    #   But A is only 2x more important than C  <- inconsistent
    #
    # Consistency Ratio (CR) measures this:
    #   CR < 0.10  -> acceptable consistency
    #   CR >= 0.10 -> matrix needs revision
    #
    # Random Index (RI) values for different matrix sizes:
    RI_dict = {
        1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12,
        6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49
    }

    n          = len(weights)
    lambda_max = float(np.mean((M @ weights) / weights))
    CI         = (lambda_max - n) / (n - 1)
    RI         = RI_dict.get(n, 1.32)
    CR         = CI / RI if RI > 0 else 0.0

    # ================================================================
    # Print Results
    # ================================================================
    if verbose:
        names = [
            "Length",
            "Diameter",
            "Confidence",
            "Tortuosity",
            "Branch Dist",
            "Edge Dist",
            "Region"
        ]

        print("\n=== AHP Weights ===")
        print(f"  {'Criterion':<15} Weight")
        print(f"  {'-'*25}")
        for name, w in zip(names, weights):
            bar = "#" * int(w * 50)  # visual bar
            print(f"  {name:<15} {w:.4f}  {bar}")

        print(f"\n  Consistency Ratio: {CR:.4f} ", end="")
        if CR < 0.10:
            print("(OK - comparisons are consistent)")
        else:
            print("(WARNING - review pairwise matrix)")

    return weights, CR