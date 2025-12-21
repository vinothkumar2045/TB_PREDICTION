import os
import random
import shutil

def balance_dataset(dataset_path):
    for split in ["train", "val", "test"]:
        normal_dir = os.path.join(dataset_path, split, "Normal")
        tb_dir = os.path.join(dataset_path, split, "Tuberculosis")

        normal_images = os.listdir(normal_dir)
        tb_images = os.listdir(tb_dir)

        normal_count = len(normal_images)
        tb_count = len(tb_images)

        print(f"--- {split.upper()} ---")
        print(f"Normal: {normal_count} images")
        print(f"Tuberculosis: {tb_count} images")

        if tb_count < normal_count:
            diff = normal_count - tb_count
            tb_to_add = random.choices(tb_images, k=diff)
            for img in tb_to_add:
                src = os.path.join(tb_dir, img)
                dst = os.path.join(tb_dir, f"copy_{random.randint(1000,9999)}_{img}")
                shutil.copy(src, dst)

        elif normal_count < tb_count:
            diff = tb_count - normal_count
            normal_to_add = random.choices(normal_images, k=diff)
            for img in normal_to_add:
                src = os.path.join(normal_dir, img)
                dst = os.path.join(normal_dir, f"copy_{random.randint(1000,9999)}_{img}")
                shutil.copy(src, dst)

    print("\n✅ Dataset balancing complete!")

dataset_path = r"C:\Users\cpvin\OneDrive\Documents\Guvi_mini_projects\TB_PREDICTION\dataset"  # path to your dataset with train/val/test
balance_dataset(dataset_path)

