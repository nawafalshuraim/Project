
import cv2
import torch
import numpy as np

img = cv2.imread("images/f1b68050c46d4876b12135f588d101b3.png", cv2.IMREAD_GRAYSCALE)
print("Image shape:", img.shape)
print("Image range:", img.min(), "->", img.max())

model = torch.jit.load("vein_model.pt", map_location="cpu")
model.eval()

img_r   = cv2.resize(img, (512, 704))
x       = torch.from_numpy(img_r.astype(np.float32)/255.0).unsqueeze(0).unsqueeze(0)

with torch.no_grad():
    out  = model(x)
    prob = torch.sigmoid(out[0,0]).numpy()

print("Prob range:", prob.min(), "->", prob.max())
print("At threshold 0.60:", (prob > 0.60).sum(), "pixels")
print("At threshold 0.65:", (prob > 0.65).sum(), "pixels")
print("At threshold 0.70:", (prob > 0.70).sum(), "pixels")
print("At threshold 0.75:", (prob > 0.75).sum(), "pixels")