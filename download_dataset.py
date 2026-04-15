from unitlab import UnitlabClient
import os
import shutil
import nest_asyncio
import inspect

nest_asyncio.apply()

API_KEY    = "ZUFoJ7CE.iyjkW0A35CZ03TQwnGsoQVQn8CRrVvCd"
DATASET_ID = "c4162f12-3e35-4514-a2ee-5edb33522d04"

client = UnitlabClient(API_KEY)
print("Client ready")
print(f"   Dataset: NIR CAMERA Dorsal Hand Vein Images")
print(f"   ID: {DATASET_ID}")

#Download annotations
print("\nDownloading annotations...")
for split in ["train", "validation", "test"]:
    try:
        client.dataset_download(DATASET_ID, "COCO", split)
        print(f"   {split}")
    except Exception as e:
        print(f"  ⚠️ {split}: {e}")

# Download images 
print("\nDownloading images...")
try:
    fn     = client.dataset_download_files
    result = fn(DATASET_ID)
    if inspect.isawaitable(result):
        import asyncio
        asyncio.get_event_loop().run_until_complete(result)
    print("Images downloaded")
except Exception as e:
    print(f"⚠️ {e}")

# Find where images were saved 
print("\nLooking for downloaded files...")
for root, dirs, files in os.walk("."):
    for f in files:
        if f.endswith((".png", ".jpg", ".jpeg")):
            print(f"  Found: {os.path.join(root, f)}")

print("\nDone!")