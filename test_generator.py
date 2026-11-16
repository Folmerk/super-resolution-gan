
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

from PIL import Image, ImageDraw

from torch.utils.data import DataLoader

from prepare_div2k import (
    OUTPUT_DIR,
    HR_PATCH_SIZE,
    SCALE,
    DIV2KDataset
)

from rrdb_generator import RRDBGenerator


# ============================================================
# 1. НАЛАШТУВАННЯ
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

RESULTS_DIR = PROJECT_DIR / "results"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 2. ПЕРЕТВОРЕННЯ TENSOR У PIL IMAGE
# ============================================================

def tensor_to_image(tensor):
    """
    Перетворює Tensor [1, 3, H, W] або [3, H, W]
    у зображення PIL RGB.
    """

    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    tensor = tensor.detach().cpu()
    tensor = tensor.clamp(0, 1)

    image_array = (
        tensor.permute(1, 2, 0).numpy() * 255
    ).round().astype(np.uint8)

    return Image.fromarray(image_array)


# ============================================================
# 3. ЗАВАНТАЖЕННЯ ВАЛІДАЦІЙНИХ ДАНИХ
# ============================================================

def load_validation_data():

    dataset = DIV2KDataset(
        root_dir=OUTPUT_DIR,
        split="valid",
        hr_patch_size=HR_PATCH_SIZE,
        scale=SCALE,
        patches_per_image=1,
        augment=False,
        degradation=False
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    return loader


# ============================================================
# 4. ПЕРЕВІРКА ГЕНЕРАТОРА
# ============================================================

def test_generator():

    print("Перевірка генератора на DIV2K")
    print("--------------------------------")

    print("Пристрій:", DEVICE)

    # Створення генератора
    generator = RRDBGenerator().to(DEVICE)

    generator.eval()

    # Завантаження валідаційного набору
    valid_loader = load_validation_data()

    # Отримуємо першу пару LR / HR
    lr, hr = next(iter(valid_loader))

    lr = lr.to(DEVICE)
    hr = hr.to(DEVICE)

    print("\nВхідні дані:")
    print("LR:", lr.shape)
    print("HR:", hr.shape)

    # Прямий прохід через генератор
    with torch.no_grad():

        sr = generator(lr)

    print("\nСтатистика виходу генератора:")
    print("Мінімальне значення:", sr.min().item())
    print("Максимальне значення:", sr.max().item())
    print("Середнє значення:", sr.mean().item())
    print("Стандартне відхилення:", sr.std().item())

    print(
        "Частка від'ємних значень:",
        (sr < 0).float().mean().item() * 100,
        "%"
    )

    print(
        "Частка значень понад 1:",
        (sr > 1).float().mean().item() * 100,
        "%"
    )
    print("\nРезультат генератора:")
    print("SR:", sr.shape)

    # ========================================================
    # 5. ПЕРЕВІРКА РОЗМІРІВ
    # ========================================================

    assert sr.shape == hr.shape, (
        "Розміри SR та HR не збігаються!"
    )

    print("\nРозміри SR та HR збігаються.")

    # ========================================================
    # 6. ПЕРЕВІРКА ЧИСЛОВИХ ЗНАЧЕНЬ
    # ========================================================

    assert torch.isfinite(sr).all(), (
        "У виході генератора є NaN або Inf!"
    )

    print("Некоректних числових значень не виявлено.")

    # ========================================================
    # 7. ПОПЕРЕДНІЙ РОЗРАХУНОК L1 LOSS
    # ========================================================

    loss = F.l1_loss(sr, hr)

    print("\nПопередній L1 Loss:", loss.item())

    # ========================================================
    # 8. ЗБЕРЕЖЕННЯ РЕЗУЛЬТАТІВ
    # ========================================================

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    lr_image = tensor_to_image(lr)
    sr_image = tensor_to_image(sr)
    hr_image = tensor_to_image(hr)

    # Збільшуємо LR для візуального порівняння
    lr_upscaled = lr_image.resize(
        hr_image.size,
        Image.Resampling.BICUBIC
    )

    width, height = hr_image.size

    # Полотно для трьох зображень
    comparison = Image.new(
        "RGB",
        (width * 3, height + 30),
        (255, 255, 255)
    )

    comparison.paste(lr_upscaled, (0, 30))
    comparison.paste(sr_image, (width, 30))
    comparison.paste(hr_image, (width * 2, 30))

    draw = ImageDraw.Draw(comparison)

    draw.text((width // 2 - 30, 8), "LR Bicubic", fill="black")
    draw.text((width + width // 2 - 15, 8), "SR", fill="black")
    draw.text((width * 2 + width // 2 - 15, 8), "HR", fill="black")

    output_path = RESULTS_DIR / "generator_test.png"

    comparison.save(output_path)

    print("\nПорівняння збережено:")
    print(output_path)

    print("\nКрок 4 успішно виконано!")


# ============================================================
# 9. ЗАПУСК
# ============================================================

if __name__ == "__main__":
    test_generator()
