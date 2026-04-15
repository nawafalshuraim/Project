# ================================================================
# Clean binary mask
#         |
#         v
# skeletonize()          <- from scikit-image (not in this file)
#         |
#         v
# Single pixel wide centerlines (but with spurs)
#         |
#         v
# skeleton_degree()
#   - counts neighbors of each pixel
#   - identifies endpoints, midpoints, junctions
#         |
#         v
# prune_spurs()
#   - removes tiny hair-like artifacts
#   - repeats 8 times
#         |
#         v
# Clean skeleton (real vein centerlines only)
#         |
#         v
# extract_graph_segments()
#   - treats skeleton as a graph
#   - finds paths between junctions/endpoints
#   - each path = one vein segment
#         |
#         v
# List of segments (each = ordered pixel path)
#   -> ready for IDSS feature extraction
# ================================================================

import numpy as np


# ================================================================
# HELPER - 8 Connected Neighbors
# ================================================================
def neighbors8(y, x, h, w):
    """
    Yield all valid 8-connected neighbors of pixel (y, x).

    8-connected means we look at all surrounding pixels:
        [top-left]  [top]  [top-right]
        [left]      (y,x)  [right]
        [bot-left]  [bot]  [bot-right]

    Stays within image bounds (0 to h, 0 to w).
    """
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue          # skip the pixel itself
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                yield ny, nx


# ================================================================
# FUNCTION 1 - Skeleton Degree
# ================================================================
def skeleton_degree(skel):
    """
    Count how many neighbors each skeleton pixel has.
    This tells us what type of pixel it is:

        degree = 1  ->  endpoint   (tip of a vein branch)
        degree = 2  ->  middle     (straight path, keep going)
        degree >= 3 ->  junction   (veins branching or crossing)

    This is used to:
      - Find endpoints for pruning spurs
      - Find junctions for segment extraction
      - Compute branch distance in IDSS

    Parameters:
        skel : (H, W) uint8 binary skeleton {0, 1}

    Returns:
        deg  : (H, W) uint8 degree count at each skeleton pixel
    """
    h, w = skel.shape
    deg  = np.zeros_like(skel, dtype=np.uint8)

    # Only process skeleton pixels (value = 1)
    ys, xs = np.where(skel > 0)

    for y, x in zip(ys, xs):
        # Count how many of the 8 neighbors are also skeleton pixels
        count = sum(
            1 for ny, nx in neighbors8(y, x, h, w)
            if skel[ny, nx]
        )
        deg[y, x] = count

    return deg


# ================================================================
# FUNCTION 2 - Prune Spurs
# ================================================================
def prune_spurs(skel, iters=8):
    """
    Remove small hair-like spurs from the skeleton.

    After skeletonization, many tiny branches appear that are
    not real veins - just artifacts of the skeletonization process.
    These are called spurs.

    How it works:
      - Find all endpoint pixels (degree = 1)
      - Delete them
      - Repeat N times
      - Each iteration removes one more pixel from each spur
      - After 8 iterations, spurs shorter than 8px are gone

    Example:
        Before:  main vein ────────┬──── spur (short)
        After:   main vein ────────┘     (spur removed)

    Parameters:
        skel  : (H, W) uint8 binary skeleton
        iters : number of pruning iterations (default 8)
                higher = removes longer spurs

    Returns:
        pruned skeleton (H, W) uint8
    """
    sk = skel.copy().astype(np.uint8)

    for _ in range(iters):
        deg       = skeleton_degree(sk)

        # Find all endpoint pixels (degree = 1)
        endpoints = (sk > 0) & (deg == 1)

        # If no endpoints left, stop early
        if endpoints.sum() == 0:
            break

        # Remove all endpoints
        sk[endpoints] = 0

    return sk


# ================================================================
# FUNCTION 3 - Extract Graph Segments
# ================================================================
def extract_graph_segments(skel, min_len_px=12):
    """
    Extract individual vein segments from the skeleton.

    Treats the skeleton as a graph:
      - Nodes    = endpoints (deg=1) and junctions (deg>=3)
      - Segments = pixel paths between two nodes

    How it works:
      1. Find all node pixels (endpoints + junctions)
      2. From each node, follow the path through deg=2 pixels
      3. Stop when reaching another node
      4. That path = one vein segment
      5. Keep only segments longer than min_len_px

    Example skeleton:
                    junction
        endpoint ───────┬─────── endpoint
                        |
                        endpoint

    Gives 3 segments:
        - left path  (endpoint to junction)
        - right path (junction to endpoint)
        - bottom path (junction to endpoint)

    Parameters:
        skel       : (H, W) uint8 pruned skeleton
        min_len_px : minimum segment length in pixels (default 12)
                     shorter segments are likely noise

    Returns:
        List of segments
        Each segment = ordered list of (y, x) tuples
        Example: [(100,200), (101,201), (102,202), ...]
    """
    h, w = skel.shape
    deg  = skeleton_degree(skel)

    # Find all node pixels
    # Nodes = endpoints (deg=1) OR junctions (deg>=3)
    nodes        = set(zip(*np.where((skel > 0) & ((deg == 1) | (deg >= 3)))))
    visited_edge = set()
    segments     = []

    def skel_neighbors(y, x):
        """Return all skeleton neighbors of (y, x)."""
        return [
            (ny, nx)
            for ny, nx in neighbors8(y, x, h, w)
            if skel[ny, nx]
        ]

    def edge_id(a, b):
        """Unique ID for an edge between two pixels."""
        return (a, b) if a <= b else (b, a)

    # Start from each node pixel
    for n in nodes:
        for nb in skel_neighbors(*n):
            eid = edge_id(n, nb)

            # Skip if already visited
            if eid in visited_edge:
                continue

            # Start a new path
            path = [n, nb]
            visited_edge.add(eid)
            prev, cur = n, nb

            # Follow the path through deg=2 pixels
            while True:
                # Stop if we reached another node
                if cur in nodes:
                    break

                # Find the next pixel (not where we came from)
                nbs = skel_neighbors(*cur)
                nxt = next((c for c in nbs if c != prev), None)

                if nxt is None:
                    break

                visited_edge.add(edge_id(cur, nxt))
                path.append(nxt)
                prev, cur = cur, nxt

            # Only keep segments long enough
            if len(path) >= min_len_px:
                segments.append(path)

    return segments