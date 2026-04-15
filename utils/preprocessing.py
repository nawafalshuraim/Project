
import cv2
import numpy as np
import torch


# FUNCTION 1 - Preprocess Image
def preprocess_image(img_path, H, W):
    """
    Load a grayscale image and prepare it for the model.

    What it does:
      1. Reads the image in grayscale
         (NIR vein images are grayscale - no color needed)
      2. Resizes to model expected dimensions (704 x 512)
      3. Normalizes pixel values from 0-255 to 0.0-1.0
      4. Converts to PyTorch tensor shape (1, 1, H, W)
         - First 1  = batch size (one image at a time)
         - Second 1 = channels (grayscale = 1 channel)
         - H, W     = height and width

    Parameters:
        img_path : path to the image file
        H        : target height (from config.py IMAGE_H = 704)
        W        : target width  (from config.py IMAGE_W = 512)

    Returns:
        img_u8  : (H, W) uint8 array      - original for visualization
        tensor  : (1, 1, H, W) float tensor - normalized for model input
    """
    # Read image in grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(
            f"Could not load image: {img_path}\n"
            f"Check the file exists and is a valid image."
        )

    # Resize to model input dimensions
    # INTER_AREA is best for shrinking images - preserves details
    img_resized = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)

    # Normalize: 0-255 pixel values -> 0.0-1.0 float values
    # Model expects normalized input matching training preprocessing
    img_float = img_resized.astype(np.float32) / 255.0

    # Convert to tensor: (H, W) -> (1, 1, H, W)
    # unsqueeze(0) adds batch dimension
    # unsqueeze(0) adds channel dimension
    tensor = torch.from_numpy(img_float).unsqueeze(0).unsqueeze(0)

    return img_resized, tensor


# ================================================================
# FUNCTION 2 - Smooth Mask
# ================================================================
def smooth_mask(mask01, close_ks=5, open_ks=3, min_area=80):
    """
    Clean up the raw binary mask from the model.

    The model output is often noisy - small dots, gaps in veins,
    fragmented regions. This function fixes all of that.

    Three cleaning steps:
      1. CLOSE  - fills small gaps inside veins
                  (connects broken vein segments)
      2. OPEN   - removes small noise speckles outside veins
                  (cleans up false positive dots)
      3. FILTER - removes tiny connected components
                  (anything smaller than min_area pixels is noise)

    Parameters:
        mask01   : (H, W) uint8 binary mask from model {0, 1}
        close_ks : kernel size for closing gaps (default 5)
        open_ks  : kernel size for removing noise (default 3)
        min_area : minimum pixel area to keep (default 80)

    Returns:
        cleaned (H, W) uint8 mask {0, 1}

    Example:
        Raw mask:     0 0 1 0 1 1 0    (broken, noisy)
        After close:  0 0 1 1 1 1 0    (gap filled)
        After open:   0 0 1 1 1 1 0    (noise removed)
        After filter: 0 0 1 1 1 1 0    (small blobs removed)
    """
    # Ensure binary values 0 or 1
    m = (mask01 > 0).astype(np.uint8)

    # Step 1 - CLOSE: fill small gaps in veins
    # Ellipse kernel works better than rectangle for curved veins
    k_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (close_ks, close_ks)
    )
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_close, iterations=1)

    # Step 2 - OPEN: remove noise speckles
    k_open = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (open_ks, open_ks)
    )
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k_open, iterations=1)

    # Step 3 - FILTER: remove small connected components
    # connectedComponentsWithStats finds all separate blobs
    # and gives us the area of each one
    num, lab, stats, _ = cv2.connectedComponentsWithStats(
        m, connectivity=8
    )

    out = np.zeros_like(m)
    for cid in range(1, num):    # start from 1 (0 = background)
        if stats[cid, cv2.CC_STAT_AREA] >= min_area:
            out[lab == cid] = 1  # keep blobs large en
    return out.astype(np.uint8)
