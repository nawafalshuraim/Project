# ================================================================
# idss/insertion.py - Insertion Point Finder
# ================================================================
# Finds the single best pixel on the best vein segment
# to place the needle for IV insertion.
#
# This is the final and most specific output of the IDSS:
# not just "use this vein segment" but
# "insert the needle exactly at this pixel coordinate"
#
# Strategy:
#   1. Ignore first 20% of segment (too close to start endpoint)
#   2. Ignore last  20% of segment (too close to end endpoint)
#   3. From the remaining 60%, find the pixel with the
#      highest distance transform value
#      = widest point of the vein
#      = most room for the needle
#      = lowest risk of puncturing vein wall
#
# Why avoid the endpoints:
#   Endpoints are where the vein branches or ends.
#   Inserting too close to an endpoint risks:
#     - Missing the vein as it narrows
#     - Hitting a junction or valve
#     - Catheter threading issues
#
# Why choose the widest point:
#   Wider vein at insertion point means:
#     - Easier needle entry
#     - Less chance of puncturing both walls
#     - More stable catheter placement
#     - Better blood flow around catheter
# ================================================================

import numpy as np


def find_insertion_point(path, dist_map):
    """
    Find the optimal needle insertion point on a vein segment.

    Parameters:
        path     : list of (y, x) tuples - ordered pixel path
                   from extract_graph_segments()

        dist_map : (H, W) float32 distance transform
                   value at each pixel = radius of vein there
                   higher value = wider vein at that point

    Returns:
        (y, x) tuple - pixel coordinates of recommended insertion point
                       y = row    (vertical position)
                       x = column (horizontal position)

    Example:
        path = [(100,200), (101,201), ..., (150,250)]
        n    = 50 pixels long

        Ignore first 20% = first 10 pixels  (too close to start)
        Ignore last  20% = last  10 pixels  (too close to end)
        Search middle 60% = pixels 10 to 40

        Find pixel with max dist_map value in that range
        -> that is the insertion point
    """
    n = len(path)

    # ================================================================
    # Step 1 - Define safe zone (middle 60% of segment)
    # ================================================================
    # Skip first 20% of path
    start = max(0, int(0.20 * n))

    # Skip last 20% of path
    end   = min(n, int(0.80 * n))

    # Extract the safe middle portion
    sub = path[start:end]

    # Safety check - if middle portion is empty use full path
    if len(sub) == 0:
        sub = path

    # ================================================================
    # Step 2 - Find widest point in safe zone
    # ================================================================
    # Get distance transform value at each pixel in safe zone
    # dist_map value = vein radius at that pixel
    # Higher value = wider vein = better insertion point
    radii = [dist_map[int(y), int(x)] for y, x in sub]

    # Find index of maximum radius
    best_idx = int(np.argmax(radii))

    # ================================================================
    # Step 3 - Return coordinates
    # ================================================================
    best_y, best_x = sub[best_idx]

    return int(best_y), int(best_x)