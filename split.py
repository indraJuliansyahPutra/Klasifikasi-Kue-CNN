import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

# Path ke dataset awal (tempat semua gambar diletakkan dalam folder per kategori)
source_dir = "dataset_raw"
target_dir = "dataset"

# Rasio split
test_size = 0.15  # 15% data untuk testing
val_size = 0.15   # 15% data untuk validation dari total data
train_size = 0.7  # 70% data untuk training

# Pastikan direktori target ada
for split in ["train", "test", "val"]:
    for category in ["kue dadar gulung", "kue kastengel", "kue klepon", "kue lapis", "kue lumpur", "kue putri salju", "kue risoles", "kue serabi"]:
        os.makedirs(os.path.join(target_dir, split, category), exist_ok=True)

# Fungsi untuk membagi dan menyalin data
for category in os.listdir(source_dir):
    category_path = os.path.join(source_dir, category)
    if not os.path.isdir(category_path):
        continue
    
    images = os.listdir(category_path)
    images = np.array(images)
    
    # Split train & temp (test + val)
    train_files, temp_files = train_test_split(images, test_size=test_size + val_size, random_state=42)
    
    # Split temp menjadi test & val
    test_files, val_files = train_test_split(temp_files, test_size=test_size / (test_size + val_size), random_state=42)
    
    # Salin file ke direktori tujuan
    for file_set, split in zip([train_files, test_files, val_files], ["train", "test", "val"]):
        for file_name in file_set:
            src_path = os.path.join(category_path, file_name)
            dst_path = os.path.join(target_dir, split, category, file_name)
            shutil.copy2(src_path, dst_path)

print("Dataset telah dibagi menjadi train (70%), test (15%), dan val (15%).")
