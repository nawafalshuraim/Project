import numpy as np
from utils.skeleton import skeleton_degree


def extract_idss_features(segments, skeleton, dist_map, prob_np, img_shape):
    """
    Extract all features for every vein segment.

    Parameters:
        segments  : list of paths from extract_graph_segments()
                    each path = list of (y, x) tuples
        skeleton  : (H, W) uint8 binary skeleton
        dist_map  : (H, W) float32 distance transform
                    value at each pixel = radius of vein there
        prob_np   : (H, W) float32 model probability map
                    value at each pixel = how confident model is
        img_shape : (H, W) tuple - image dimensions

    Returns:
        List of feature dicts, one per segment
        Each dict contains all 10 features + raw path
    """
    H, W = img_shape

    # Find all junction pixels (degree >= 3)
    # These are where veins branch or cross
    # Used to compute branch distances
    deg             = skeleton_degree(skeleton)
    junction_mask   = (skeleton > 0) & (deg >= 3)
    junction_coords = np.array(np.where(junction_mask)).T  # (N, 2) [y, x]

    # Define anatomical regions of the dorsal hand
    # Based on vertical position in image (y coordinate)
    #
    #   Top 40%    = Distal   (fingers/knuckles) -> preferred for IV
    #   40% - 70%  = Middle   (metacarpal area)  -> acceptable
    #   Bottom 30% = Proximal (wrist area)        -> avoid (joints)
    #
    DISTAL_Y_MAX = int(0.40 * H)   # top 40%
    MIDDLE_Y_MAX = int(0.70 * H)   # top 70%
    WRIST_Y_MIN  = int(0.80 * H)   # bottom 20% = wrist = reject

    features = []

    for path in segments:
        coords = np.array(path, dtype=np.float32)  # (N, 2) [y, x]

        # ── Feature 1: Length ─────────────────────────────────
        # Geodesic length = actual path length along the vein
        # Not straight-line distance - follows the vein curve
        # Longer veins give more room for needle insertion
        diffs     = coords[1:] - coords[:-1]
        length_px = float(np.sqrt((diffs**2).sum(axis=1)).sum())

        # ── Feature 2: Diameter ───────────────────────────────
        # Distance transform value = radius at that pixel
        # Diameter = 2 x mean radius along the segment
        # Larger diameter = easier to insert catheter
        radii       = [dist_map[int(y), int(x)] for y, x in path]
        diameter_px = float(2.0 * np.mean(radii)) if radii else 0.0

        # ── Feature 3: Confidence ─────────────────────────────
        # Mean model probability along the segment
        # Higher = model is more sure this is a real vein
        conf_vals  = [prob_np[int(y), int(x)] for y, x in path]
        confidence = float(np.mean(conf_vals)) if conf_vals else 0.0

        # ── Feature 4: Confidence Variation ───────────────────
        # How much confidence varies along the segment
        # Spec: must not vary more than 15% across segment length
        # High variation = weak spot somewhere along the vein
        #
        # Example:
        #   Good: 0.95, 0.97, 0.96, 0.95  -> variation = 0.02 (2%)
        #   Bad:  0.99, 0.95, 0.71, 0.98  -> variation = 0.28 (28%)
        conf_variation = float(
            max(conf_vals) - min(conf_vals)
        ) if conf_vals else 1.0

        # ── Feature 5: Tortuosity ─────────────────────────────
        # How curved/winding the vein is
        # Calculated as: 1 - (straight line / actual path length)
        #
        #   0.0 = perfectly straight
        #   1.0 = extremely curved
        #
        # Straight veins are easier and safer for needle insertion
        y0, x0     = coords[0]
        y1, x1     = coords[-1]
        euclid     = float(np.hypot(y1 - y0, x1 - x0))
        tortuosity = 1.0 - min(1.0, euclid / (length_px + 1e-6))

        # ── Feature 6: Branch Distance ────────────────────────
        # Distance from segment center to nearest junction pixel
        # Far from branches = safer insertion zone
        if len(junction_coords) > 0:
            seg_mean_y = float(np.mean(coords[:, 0]))
            seg_mean_x = float(np.mean(coords[:, 1]))
            dists      = np.sqrt(
                (junction_coords[:, 0] - seg_mean_y)**2 +
                (junction_coords[:, 1] - seg_mean_x)**2
            )
            branch_distance = float(dists.min())
        else:
            branch_distance = float(W)  # no junctions = maximum distance

        # ── Feature 7 & 8: Endpoint Branch Distances ──────────
        # Spec: each endpoint must be at least 3mm (~15px)
        #       away from any bifurcation or crossing
        #
        # We check both ends of the segment separately
        # because one end might be safe while the other is not
        if len(junction_coords) > 0:
            # Start endpoint
            sy = int(coords[0][0])
            sx = int(coords[0][1])
            endpoint_dist_start = float(np.sqrt(
                (junction_coords[:, 0] - sy)**2 +
                (junction_coords[:, 1] - sx)**2
            ).min())

            # End endpoint
            ey = int(coords[-1][0])
            ex = int(coords[-1][1])
            endpoint_dist_end = float(np.sqrt(
                (junction_coords[:, 0] - ey)**2 +
                (junction_coords[:, 1] - ex)**2
            ).min())
        else:
            endpoint_dist_start = float(W)  # no junctions = safe
            endpoint_dist_end   = float(W)

        # ── Feature 9: Edge Distance ──────────────────────────
        # Mean distance from segment pixels to image border
        # Veins too close to edge are unstable for insertion
        # (skin folds, image artifacts near borders)
        edge_dists    = [
            min(int(y), H - int(y), int(x), W - int(x))
            for y, x in path
        ]
        edge_distance = float(np.mean(edge_dists))

        # ── Feature 10: Region Score ──────────────────────────
        # Anatomical location score based on vertical position
        # Distal (fingers) preferred, wrist avoided
        mean_y = float(np.mean(coords[:, 0]))
        mean_x = float(np.mean(coords[:, 1]))

        if mean_y <= DISTAL_Y_MAX:
            region_score = 1.0    # distal = best
        elif mean_y <= MIDDLE_Y_MAX:
            region_score = 0.6    # middle = acceptable
        else:
            region_score = 0.1    # proximal = avoid

        # Wrist flag = hard reject if over wrist joint
        wrist_flag = mean_y >= WRIST_Y_MIN

        # Collect all features for this segment
        features.append({
            "path":                path,
            "length_px":           length_px,
            "diameter_px":         diameter_px,
            "confidence":          confidence,
            "conf_variation":      conf_variation,
            "tortuosity":          tortuosity,
            "branch_distance":     branch_distance,
            "endpoint_dist_start": endpoint_dist_start,
            "endpoint_dist_end":   endpoint_dist_end,
            "edge_distance":       edge_distance,
            "region_score":        region_score,
            "mean_y":              mean_y,
            "mean_x":              mean_x,
            "wrist_flag":          wrist_flag,
        })

    return features