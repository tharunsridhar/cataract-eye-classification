from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from PIL import Image

from cataract_classifier.config_loader import load_config


def check_image_status(directory) -> None:
    if not os.path.isdir(directory):
        print(f"Error: Directory not found at {directory}")
        return

    print(f"Checking images in directory: {directory}")
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            try:
                img = Image.open(filepath)
                img.verify()
                print(f"{filename}: NOT corrupted")
            except (OSError, SyntaxError) as exc:
                print(f"{filename}: IS corrupted - {exc}")
            except Exception as exc:
                print(f"{filename}: Could not process - {exc}")
        else:
            print(f"{filename}: IS NOT a file (skipping)")


def check_folder_names(base_directory, expected_folders) -> None:
    print(f"Checking folder names in base directory: {base_directory}")
    if not os.path.isdir(base_directory):
        print(f"Error: Base directory not found at {base_directory}")
        return

    found_folders = {
        item for item in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, item))
    }
    expected_folders_set = set(expected_folders)
    missing_folders = expected_folders_set - found_folders
    unexpected_folders = found_folders - expected_folders_set

    if missing_folders:
        print("Warning: Expected folders not found:")
        for folder in missing_folders:
            print(f"- {folder}")
    if unexpected_folders:
        print("Warning: Unexpected folders found:")
        for folder in unexpected_folders:
            print(f"- {folder}")
    if not missing_folders and not unexpected_folders:
        print("All expected folders found and no unexpected folders.")


def check_image_files_only(directory) -> None:
    print(f"Checking files in directory: {directory}")
    if not os.path.isdir(directory):
        print(f"Error: Directory not found at {directory}")
        return

    non_image_files_found = False
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                print(f"Warning: Non-image file found: {filename}")
                non_image_files_found = True
        else:
            print(f"Info: Non-file item found (skipping): {filename}")
    if not non_image_files_found:
        print("All files appear to be images.")


def check_filename_label_accuracy(base_directory, class_es, sub_class_es) -> None:
    print(f"Checking filename label accuracy in base directory: {base_directory}")
    if not os.path.isdir(base_directory):
        print(f"Error: Base directory not found at {base_directory}")
        return

    for cls in class_es:
        class_folder_path = os.path.join(base_directory, cls)
        if not os.path.isdir(class_folder_path):
            print(f"Warning: Class folder not found (skipping): {class_folder_path}")
            continue
        for sub_cls in sub_class_es:
            sub_class_folder_path = os.path.join(class_folder_path, sub_cls)
            print(f"\nChecking folder: {sub_class_folder_path}")
            if not os.path.isdir(sub_class_folder_path):
                print(f"Warning: Sub-class folder not found (skipping): {sub_class_folder_path}")
                continue
            for filename in os.listdir(sub_class_folder_path):
                filepath = os.path.join(sub_class_folder_path, filename)
                if os.path.isfile(filepath):
                    if sub_cls.lower() not in filename.lower():
                        print(
                            f"Warning: Filename '{filename}' in folder '{sub_cls}' "
                            "does not contain the folder name."
                        )
                    else:
                        print(f"Filename '{filename}' in folder '{sub_cls}' contains the folder name.")
                else:
                    print(f"Info: Non-file item found (skipping): {filename}")


def check_image_size_consistency(directory, expected_size) -> None:
    print(f"Checking image sizes in directory: {directory}")
    if not os.path.isdir(directory):
        print(f"Error: Directory not found at {directory}")
        return

    size_mismatch_found = False
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            try:
                img = Image.open(filepath)
                if img.size != tuple(expected_size):
                    print(f"Warning: Image '{filename}' has size {img.size}, expected {expected_size}.")
                    size_mismatch_found = True
            except OSError:
                print(f"Error: Could not open or process image file: {filename}")
                size_mismatch_found = True
        else:
            print(f"Info: Non-file item found (skipping): {filename}")
    if not size_mismatch_found:
        print(f"All images in directory '{directory}' are of expected size {expected_size}.")


def check_class_distribution(base_directory, class_es, sub_class_es) -> pd.DataFrame:
    print(f"Checking class distribution in base directory: {base_directory}")
    if not os.path.isdir(base_directory):
        print(f"Error: Base directory not found at {base_directory}")
        return pd.DataFrame()

    distribution_data = {}
    total_images = 0
    for cls in class_es:
        class_folder_path = os.path.join(base_directory, cls)
        if not os.path.isdir(class_folder_path):
            print(f"Warning: Class folder not found (skipping): {class_folder_path}")
            continue
        distribution_data[cls] = {}
        class_total = 0
        for sub_cls in sub_class_es:
            sub_class_folder_path = os.path.join(class_folder_path, sub_cls)
            if not os.path.isdir(sub_class_folder_path):
                print(f"Warning: Sub-class folder not found (skipping): {sub_class_folder_path}")
                distribution_data[cls][sub_cls] = 0
                continue
            image_count = sum(
                1
                for filename in os.listdir(sub_class_folder_path)
                if os.path.isfile(os.path.join(sub_class_folder_path, filename))
                and filename.lower().endswith((".png", ".jpg", ".jpeg"))
            )
            distribution_data[cls][sub_cls] = image_count
            class_total += image_count
        distribution_data[cls]["Total"] = class_total
        total_images += class_total

    df_distribution = pd.DataFrame(distribution_data)
    if not df_distribution.empty:
        df_distribution["Sub-class Total"] = df_distribution.loc[sub_class_es].sum(axis=1)
    print("\nImage Distribution per Sub-class and Class:")
    print(df_distribution)
    print(f"\nTotal number of images found: {total_images}")
    return df_distribution


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data_config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    split_names = list(cfg["data_dir"].keys())
    class_names = cfg["class_names"]
    base_dir = Path(cfg["data_dir"]["train"]).parent
    check_folder_names(base_dir, split_names)
    for split in split_names:
        check_folder_names(base_dir / split, class_names)
        for class_name in class_names:
            folder = base_dir / split / class_name
            check_image_status(folder)
            check_image_files_only(folder)
            check_image_size_consistency(folder, cfg["image_size"])
    check_filename_label_accuracy(base_dir, split_names, class_names)
    check_class_distribution(base_dir, split_names, class_names)


if __name__ == "__main__":
    main()
