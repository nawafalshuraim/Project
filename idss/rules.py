MIN_LENGTH_PX            = 30     # Rule 3:  min vein length for IV
MIN_DIAMETER_PX          = 4      # Rule 2:  min diameter for catheter
MAX_TORTUOSITY           = 0.45   # Rule 9:  max curvature allowed
MIN_BRANCH_DIST_PX       = 15     # Rule 4:  min dist from branch point
MIN_EDGE_DIST_PX         = 20     # Rule 8:  min dist from image border
MIN_CONFIDENCE           = 0.70   # Rule 11: min model confidence
MIN_USABLE_LENGTH        = 25     # Rule 18: min usable length
MAX_CONF_VARIATION       = 0.15   # Spec:    max 15% confidence variation
MIN_ENDPOINT_BRANCH_DIST = 15     # Spec:    endpoints >= 3mm from bifurcation


def apply_knowledge_rules(feat):
    """
    Apply all clinical rules to one segment.

    Parameters:
        feat : feature dict from extract_idss_features()

    Returns:
        accepted : bool  - False means segment is rejected entirely
        penalty  : float - multiplied into final score (reduces score)
        bonus    : float - multiplied into final score (increases score)
        reasons  : list  - explanation of every rule that fired
    """
    accepted = True
    penalty  = 1.0
    bonus    = 1.0
    reasons  = []

    # ================================================================
    # HARD REJECTION RULES
    # ================================================================
    # These rules completely remove a segment from consideration.
    # If any rejection fires, we return immediately.
    # No point checking other rules on a rejected segment.

    # Rule 3 - Segment Too Short
    # A vein segment must be long enough to insert a needle safely.
    # Too short = no room to maneuver, high risk of missing.
    if feat["length_px"] < MIN_LENGTH_PX:
        accepted = False
        reasons.append(
            f"REJECTED: too short "
            f"({feat['length_px']:.1f}px < {MIN_LENGTH_PX}px)"
        )
        return accepted, penalty, bonus, reasons

    # Rule 2 - Diameter Too Small
    # Vein must be wide enough to accommodate the catheter.
    # Too thin = catheter will not fit, high chance of failure.
    if feat["diameter_px"] < MIN_DIAMETER_PX:
        accepted = False
        reasons.append(
            f"REJECTED: too thin "
            f"({feat['diameter_px']:.1f}px < {MIN_DIAMETER_PX}px)"
        )
        return accepted, penalty, bonus, reasons

    # Rule 6 - Over Wrist Joint
    # Veins over the wrist joint are rejected.
    # Inserting over a joint causes pain and high failure rate.
    # Patient movement at the joint dislodges the catheter.
    if feat["wrist_flag"]:
        accepted = False
        reasons.append("REJECTED: segment over wrist joint region")
        return accepted, penalty, bonus, reasons

    # Rule 11 - Low Model Confidence
    # If the model is not sure this is a real vein, reject it.
    # Low confidence means the detection may be unreliable.
    if feat["confidence"] < MIN_CONFIDENCE:
        accepted = False
        reasons.append(
            f"REJECTED: low model confidence "
            f"({feat['confidence']:.3f} < {MIN_CONFIDENCE})"
        )
        return accepted, penalty, bonus, reasons

    # Rule 18 - Usable Length Too Short
    # Even if total length is ok, usable insertion length
    # after accounting for endpoints must be sufficient.
    if feat["length_px"] < MIN_USABLE_LENGTH:
        accepted = False
        reasons.append(
            f"REJECTED: usable length too short "
            f"({feat['length_px']:.1f}px < {MIN_USABLE_LENGTH}px)"
        )
        return accepted, penalty, bonus, reasons

    # Specification - Confidence Variation
    # Confidence must not vary more than 15% along the segment.
    # High variation means there is a weak/uncertain spot on the vein.
    # Inserting at a weak spot risks missing the vein entirely.
    #
    # Example:
    #   Pass: confidence goes 0.95 -> 0.97 -> 0.96  (variation = 2%)
    #   Fail: confidence goes 0.99 -> 0.71 -> 0.98  (variation = 28%)
    if feat["conf_variation"] > MAX_CONF_VARIATION:
        accepted = False
        reasons.append(
            f"REJECTED: confidence varies "
            f"{feat['conf_variation']*100:.1f}% along segment "
            f"(max allowed: 15%)"
        )
        return accepted, penalty, bonus, reasons

    # ================================================================
    # SOFT PENALTY RULES
    # ================================================================
    # These rules reduce the score but do not reject the segment.
    # Multiple penalties stack by multiplication.
    # Example: two penalties of 0.7 gives 0.7 x 0.7 = 0.49

    # Rule 4 - Close to Branch Point
    # Inserting near a branch point risks puncturing the wrong vessel.
    # Also makes it harder to thread the catheter correctly.
    if feat["branch_distance"] < MIN_BRANCH_DIST_PX:
        penalty *= 0.60
        reasons.append(
            f"PENALTY: segment center close to branch "
            f"({feat['branch_distance']:.1f}px < {MIN_BRANCH_DIST_PX}px)"
        )

    # Rule 8 - Close to Image Edge
    # Segments near the image border may be partially cut off.
    # Also indicates vein is near skin fold or edge of hand.
    if feat["edge_distance"] < MIN_EDGE_DIST_PX:
        penalty *= 0.70
        reasons.append(
            f"PENALTY: close to image edge "
            f"({feat['edge_distance']:.1f}px < {MIN_EDGE_DIST_PX}px)"
        )

    # Rule 9 - High Tortuosity
    # Very curved veins are harder to insert needle into.
    # Catheter threading is difficult in winding vessels.
    if feat["tortuosity"] > MAX_TORTUOSITY:
        penalty *= 0.65
        reasons.append(
            f"PENALTY: high tortuosity "
            f"({feat['tortuosity']:.3f} > {MAX_TORTUOSITY})"
        )

    # Specification - Start Endpoint Near Bifurcation
    # Spec: each endpoint must be at least 3mm (~15px)
    # from any bifurcation or crossing.
    # Start endpoint check.
    if feat["endpoint_dist_start"] < MIN_ENDPOINT_BRANCH_DIST:
        penalty *= 0.70
        reasons.append(
            f"PENALTY: start endpoint near bifurcation "
            f"({feat['endpoint_dist_start']:.1f}px "
            f"< {MIN_ENDPOINT_BRANCH_DIST}px)"
        )

    # Specification - End Endpoint Near Bifurcation
    # End endpoint check.
    if feat["endpoint_dist_end"] < MIN_ENDPOINT_BRANCH_DIST:
        penalty *= 0.70
        reasons.append(
            f"PENALTY: end endpoint near bifurcation "
            f"({feat['endpoint_dist_end']:.1f}px "
            f"< {MIN_ENDPOINT_BRANCH_DIST}px)"
        )

    # ================================================================
    # SOFT BONUS RULES
    # ================================================================
    # These rules increase the score for desirable properties.
    # Multiple bonuses also stack by multiplication.

    # Rule 14 - Far From Branches
    # Segment center is well away from any junction.
    # More room to insert without risk of hitting a branch.
    if feat["branch_distance"] > MIN_BRANCH_DIST_PX * 3:
        bonus *= 1.15
        reasons.append("BONUS: far from branch points")

    # Rule 1 - Very Straight Vein
    # Low tortuosity means easy, predictable needle path.
    # Straight veins have the highest insertion success rate.
    if feat["tortuosity"] < 0.10:
        bonus *= 1.10
        reasons.append(
            f"BONUS: very straight vein "
            f"(tortuosity = {feat['tortuosity']:.3f})"
        )

    # Rule 7 and 15 - Anatomical Region
    # Distal dorsal hand is the preferred IV insertion site.
    # Middle is acceptable. Proximal/wrist is penalized.
    if feat["region_score"] >= 1.0:
        bonus *= 1.20
        reasons.append("BONUS: distal dorsal hand region (preferred)")
    elif feat["region_score"] >= 0.6:
        bonus *= 1.05
        reasons.append("BONUS: middle hand region (acceptable)")
    else:
        penalty *= 0.80
        reasons.append("PENALTY: proximal region (near wrist)")

    # Rule 17 - High Model Confidence
    # Model is very certain this is a real vein.
    # High confidence = reliable detection = safer insertion.
    if feat["confidence"] >= 0.95:
        bonus *= 1.10
        reasons.append(
            f"BONUS: high model confidence "
            f"({feat['confidence']:.3f})"
        )

    # Rule 13 - Long Continuous Segment
    # Longer segments give more options for insertion point.
    # Also indicates a more stable, well-defined vessel.
    if feat["length_px"] >= 100:
        bonus *= 1.10
        reasons.append(
            f"BONUS: long continuous segment "
            f"({feat['length_px']:.1f}px)"
        )

    # Specification - Both Endpoints Safe
    # Both endpoints are well away from any bifurcation.
    # Extra confidence that the entire segment is safe.
    if (feat["endpoint_dist_start"] >= MIN_ENDPOINT_BRANCH_DIST * 2 and
            feat["endpoint_dist_end"] >= MIN_ENDPOINT_BRANCH_DIST * 2):
        bonus *= 1.10
        reasons.append("BONUS: both endpoints safely away from bifurcations")

    return accepted, penalty, bonus, reasons