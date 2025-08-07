import cv2, glob, os

# 1) Positive angle to rotate clockwise.
ANGLE_DEG = 4.0  

# 2) Paths (adjust if needed)
SRC_FOLDER = 'data/images'         # your extracted frames
DST_FOLDER = 'data/images_deskew'  # your deskewed output

os.makedirs(DST_FOLDER, exist_ok=True)

# 3) Gather files
patterns = [f'{SRC_FOLDER}/*.jpg', f'{SRC_FOLDER}/*.png']
files = []
for p in patterns:
    files.extend(glob.glob(p))
print(f'Found {len(files)} frames to deskew.')

# 4) Apply rotation
for src in files:
    img = cv2.imread(src)
    if img is None:
        continue
    h, w = img.shape[:2]
    # Use +ANGLE_DEG to rotate frames clockwise
    M = cv2.getRotationMatrix2D((w/2, h/2), +ANGLE_DEG, 1.0)
    deskewed = cv2.warpAffine(img, M, (w, h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
    dst = os.path.join(DST_FOLDER, os.path.basename(src))
    cv2.imwrite(dst, deskewed)

print(f'Deskew complete → {DST_FOLDER} (rotated +{ANGLE_DEG}° clockwise)')

