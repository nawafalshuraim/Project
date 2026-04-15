

# Model 
MODEL_PATH     = "vein_model.pt"   # change this when you get a new model
MODEL_CHANNELS = 1                 # 1 = segmentation only, 2 = segmentation + SDF

# Image dimensions (must match training dimensions) 
IMAGE_H = 704
IMAGE_W = 512

# Detection 
THRESHOLD = 0.65    # probability cutoff: above = vein, below = background

# Skeleton 
MIN_SEGMENT_LEN  = 12   # minimum segment length in pixels to keep
SPUR_PRUNE_ITERS = 8    # how many times to prune tiny branches

# IDSS Clinical Thresholds 
MIN_LENGTH_PX            = 30     # minimum vein length for IV insertion
MIN_DIAMETER_PX          = 4      # minimum vein diameter for catheter
MAX_TORTUOSITY           = 0.45   # maximum allowed curvature
MIN_BRANCH_DIST_PX       = 15     # minimum distance from branching point
MIN_EDGE_DIST_PX         = 20     # minimum distance from image border
MIN_CONFIDENCE           = 0.60   # minimum model confidence
MIN_USABLE_LENGTH        = 25     # minimum usable segment length
MAX_CONF_VARIATION       = 0.15   # max confidence variation along segment (15%)
MIN_ENDPOINT_BRANCH_DIST = 15     # min endpoint distance from bifurcation (~3mm)

# Video optimization
REANALYZE_EVERY = 3   # run IDSS every 3 frames 