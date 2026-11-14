import os
import random
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

# Налаштування шляхів та параметрів
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PATCH_SIZE = 224
MAX_SOLAR_IMAGES = 450      # Скільки фото з сонячними панелями беремо для балансу
PATCHES_PER_DIV2K = 3       # Скільки квадратів 224x224 вирізаємо з одного фото DIV2K


def create_dirs():
    """Створює структуру папок для train та val вибірок."""
    for split in ["train", "val"]:
        for sub in ["HR", "LR_x2", "LR_x4"]:
            (PROCESSED_DIR / split / sub).mkdir(parents=True, exist_ok=True)


def save_image_triplets(hr_img: Image.Image, save_name: str, split: str):
    """Зберігає еталон HR (224x224) та його зменшені бікубічні копії LR_x2 і LR_x4."""
    hr_path = PROCESSED_DIR / split / "HR" / f"{save_name}.png"
    lr_x2_path = PROCESSED_DIR / split / "LR_x2" / f"{save_name}.png"
    lr_x4_path = PROCESSED_DIR / split / "LR_x4" / f"{save_name}.png"

    # Збереження еталону High Resolution
    hr_img.save(hr_path, format="PNG")

    # Бікубічне зменшення у 2 рази (112x112)
    lr_x2 = hr_img.resize((PATCH_SIZE // 2, PATCH_SIZE // 2), Image.Resampling.BICUBIC)
    lr_x2.save(lr_x2_path, format="PNG")

    # Бікубічне зменшення у 4 рази (56x56)
    lr_x4 = hr_img.resize((PATCH_SIZE // 4, PATCH_SIZE // 4), Image.Resampling.BICUBIC)
    lr_x4.save(lr_x4_path, format="PNG")


def process_solar_dataset():
    """Відбирає фото за масками, де реально є сонячні панелі."""
    input_dir = RAW_DIR / "dutch_solar" / "input"
    mask_dir = RAW_DIR / "dutch_solar" / "mask"

    if not input_dir.exists() or not mask_dir.exists():
        print("Папка dutch_solar не знайдена, пропускаємо...")
        return

    valid_files = []
    print("Аналіз масок сонячних панелей...")
    for mask_file in sorted(mask_dir.glob("*.png")):
        mask_arr = np.array(Image.open(mask_file))
        # Якщо на масці є хоча б один білий піксель (панель присутня)
        if np.sum(mask_arr) > 0:
            img_file = input_dir / mask_file.name
            if img_file.exists():
                valid_files.append(img_file)

    random.shuffle(valid_files)
    selected_files = valid_files[:MAX_SOLAR_IMAGES]

    # Розподіл 85% у train, 15% у val
    split_idx = int(len(selected_files) * 0.85)
    train_files = selected_files[:split_idx]
    val_files = selected_files[split_idx:]

    for idx, img_path in enumerate(tqdm(train_files, desc="Solar Train")):
        img = Image.open(img_path).convert("RGB")
        if img.size == (PATCH_SIZE, PATCH_SIZE):
            save_image_triplets(img, f"solar_{idx:04d}", "train")

    for idx, img_path in enumerate(tqdm(val_files, desc="Solar Val")):
        img = Image.open(img_path).convert("RGB")
        if img.size == (PATCH_SIZE, PATCH_SIZE):
            save_image_triplets(img, f"solar_val_{idx:04d}", "val")


def process_div2k_dataset():
    """Нарізає великі зображення DIV2K на патчі 224x224."""
    for folder_name, split in [("DIV2K_train_HR", "train"), ("DIV2K_valid_HR", "val")]:
        src_dir = RAW_DIR / "div2k" / folder_name
        if not src_dir.exists():
            print(f"Папка {folder_name} не знайдена, пропускаємо...")
            continue

        img_files = sorted(src_dir.glob("*.png"))
        patch_counter = 0

        for img_path in tqdm(img_files, desc=f"DIV2K {split}"):
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            if w < PATCH_SIZE or h < PATCH_SIZE:
                continue

            # Для валідації беремо 1 центральний фрагмент, для навчання — кілька випадкових
            num_crops = PATCHES_PER_DIV2K if split == "train" else 1
            for _ in range(num_crops):
                x = random.randint(0, w - PATCH_SIZE)
                y = random.randint(0, h - PATCH_SIZE)
                patch = img.crop((x, y, x + PATCH_SIZE, y + PATCH_SIZE))
                save_image_triplets(patch, f"div2k_{split}_{patch_counter:05d}", split)
                patch_counter += 1


if __name__ == "__main__":
    random.seed(42)
    create_dirs()
    process_solar_dataset()
    process_div2k_dataset()
    print("Готово! Всі пари HR / LR_x2 / LR_x4 успішно згенеровано у data/processed/")