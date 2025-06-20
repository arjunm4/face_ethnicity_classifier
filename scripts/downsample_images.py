import os
import shutil
import random

SRC_DIR = 'data/train'
DST_DIR = 'data/train_small'
IMAGES_PER_CLASS = 1500  # or whatever you want

os.makedirs(DST_DIR, exist_ok=True)

for class_name in os.listdir(SRC_DIR):
    class_path = os.path.join(SRC_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    selected = random.sample(images, min(IMAGES_PER_CLASS, len(images)))

    dst_class_path = os.path.join(DST_DIR, class_name)
    os.makedirs(dst_class_path, exist_ok=True)

    for img in selected:
        src = os.path.join(class_path, img)
        dst = os.path.join(dst_class_path, img)
        shutil.copy2(src, dst)

print("✅ Downsampled images copied to data/train_small/")
