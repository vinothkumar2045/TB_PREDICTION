import os
import shutil
import random

# Path to your dataset with Normal/ and Tuberculosis/ folders
SOURCE_DIR = r"C:\Users\cpvin\Downloads\TB_DATA\TB_Chest_Radiography_Database"
OUTPUT_DIR = r"C:\Users\cpvin\Downloads\dataset"

# Split ratios
train_split = 0.7
val_split = 0.15
test_split = 0.15

# Create output folders
for split in ['train', 'val', 'test']:
    for category in ['Normal', 'Tuberculosis']:
        os.makedirs(os.path.join(OUTPUT_DIR, split, category), exist_ok=True)

# Function to split one category
def split_category(category):
    src_folder = os.path.join(SOURCE_DIR, category)
    images = os.listdir(src_folder)
    random.shuffle(images)

    train_end = int(train_split * len(images))
    val_end = train_end + int(val_split * len(images))

    for i, img in enumerate(images):
        src_path = os.path.join(src_folder, img)
        if i < train_end:
            dst_path = os.path.join(OUTPUT_DIR, 'train', category, img)
        elif i < val_end:
            dst_path = os.path.join(OUTPUT_DIR, 'val', category, img)
        else:
            dst_path = os.path.join(OUTPUT_DIR, 'test', category, img)
        shutil.copy2(src_path, dst_path)

# Run for both categories
split_category('Normal')
split_category('Tuberculosis')

print("✅ Data split complete!")
print(f"Train, Val, Test folders saved in: {OUTPUT_DIR}")
