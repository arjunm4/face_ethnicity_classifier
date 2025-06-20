import os
import shutil
import pandas as pd

def sort_images(csv_path, image_dir, output_dir):
    df = pd.read_csv(csv_path, encoding='utf-8-sig')

    for _, row in df.iterrows():
        file_name = row['file'] 
        race = row['race'].strip()

        src = os.path.join(image_dir, file_name)  # e.g. data/fairface-img-margin025-trainval/train/1.jpg
        dst_dir = os.path.join(output_dir, race)  # e.g. data/train/East Asian/
        dst = os.path.join(dst_dir, os.path.basename(file_name))  # e.g. data/train/East Asian/1.jpg

        os.makedirs(dst_dir, exist_ok=True)

        if os.path.exists(src):
            shutil.copy(src, dst)
        else:
            print(f"⚠️ File not found: {src}")


if __name__ == "__main__":
    base_dir = "data"
    image_dir = os.path.join(base_dir, "fairface-img-margin025-trainval")

    sort_images(os.path.join(base_dir, "fairface_label_train.csv"), image_dir, os.path.join(base_dir, "train"))
    sort_images(os.path.join(base_dir, "fairface_label_val.csv"), image_dir, os.path.join(base_dir, "val"))
