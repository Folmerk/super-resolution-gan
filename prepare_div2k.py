import io
from pathlib import Path
from PIL import Image, ImageOps, ImageFilter
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ============================================================
# НАЛАШТУВАННЯ
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

RAW_DIR = PROJECT_DIR / "data" / "raw" / "div2k"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed" / "div2k"

TRAIN_HR_DIR = RAW_DIR / "DIV2K_train_HR"
VALID_HR_DIR = RAW_DIR / "DIV2K_valid_HR"

SCALE = 4
HR_PATCH_SIZE = 256
LR_PATCH_SIZE = HR_PATCH_SIZE // SCALE

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# ============================================================
# 1. ПЕРЕВІРКА ТА ОБРОБКА ЗОБРАЖЕНЬ
# ============================================================

def get_image_files(folder):
    """Повертає список підтримуваних зображень у папці."""

    if not folder.exists():
        raise FileNotFoundError(f"Папку не знайдено: {folder}")

    return sorted(
        file for file in folder.iterdir()
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    )


def inspect_image(path):
    """
    Перевіряє цілісність зображення та можливість його декодування.

    Повертає:
    - ширину та висоту;
    - кількість каналів;
    - режим зображення.
    """

    try:
        # Перевірка цілісності файлу
        with Image.open(path) as image:
            image.verify()

        # Повторне відкриття для повного декодування
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)
            image.load()

            width, height = image.size
            channels = len(image.getbands())
            mode = image.mode

        return {
            "valid": True,
            "width": width,
            "height": height,
            "channels": channels,
            "mode": mode,
            "error": None
        }

    except Exception as error:
        return {
            "valid": False,
            "width": None,
            "height": None,
            "channels": None,
            "mode": None,
            "error": str(error)
        }


def prepare_image(input_path, output_hr_dir, output_lr_dir):
    """
    1. Відкриває HR-зображення.
    2. Перетворює його у RGB.
    3. Обрізає розміри до кратних SCALE.
    4. Створює LR-зображення методом Bicubic.
    5. Зберігає HR та LR у відповідні папки.
    """

    with Image.open(input_path) as image:
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")

        width, height = image.size

        # Обрізання правого та нижнього країв
        new_width = (width // SCALE) * SCALE
        new_height = (height // SCALE) * SCALE

        if new_width == 0 or new_height == 0:
            raise ValueError(
                f"Зображення замале для масштабу ×{SCALE}: {input_path}"
            )

        image = image.crop((0, 0, new_width, new_height))

        # Створення LR зі зменшенням у 4 рази
        lr_width = new_width // SCALE
        lr_height = new_height // SCALE

        lr_image = image.resize(
            (lr_width, lr_height),
            Image.Resampling.BICUBIC
        )

        # Зберігаємо у PNG без втрат
        output_name = input_path.stem + ".png"

        image.save(output_hr_dir / output_name, format="PNG")
        lr_image.save(output_lr_dir / output_name, format="PNG")

        return {
            "original_size": (width, height),
            "hr_size": (new_width, new_height),
            "lr_size": (lr_width, lr_height)
        }


def apply_degradations(lr_image):
    """
    Випадково погіршує якість LR-зображення.

    Можливі деградації:
    1. Gaussian Blur — розмиття.
    2. Gaussian Noise — додавання шуму.
    3. JPEG Compression — стиснення з втратами.

    На вхід отримує PIL RGB Image.
    Повертає PIL RGB Image.
    """

    # --------------------------------------------------------
    # 1. Gaussian Blur — розмиття
    # --------------------------------------------------------

    if random.random() < 0.3:
        radius = random.uniform(0.1, 1.5)

        lr_image = lr_image.filter(
            ImageFilter.GaussianBlur(radius=radius)
        )

    # --------------------------------------------------------
    # 2. Gaussian Noise — додавання шуму
    # --------------------------------------------------------

    if random.random() < 0.3:
        image_array = np.asarray(
            lr_image,
            dtype=np.float32
        )

        # Випадкова інтенсивність шуму
        sigma = random.uniform(1.0, 15.0)

        noise = np.random.normal(
            loc=0.0,
            scale=sigma,
            size=image_array.shape
        )

        image_array = image_array + noise

        # Обмеження значень пікселів діапазоном [0, 255]
        image_array = np.clip(
            image_array,
            0,
            255
        ).astype(np.uint8)

        lr_image = Image.fromarray(image_array)

    # --------------------------------------------------------
    # 3. JPEG Compression — стиснення з втратами
    # --------------------------------------------------------

    if random.random() < 0.3:
        quality = random.randint(30, 100)

        buffer = io.BytesIO()

        lr_image.save(
            buffer,
            format="JPEG",
            quality=quality
        )

        buffer.seek(0)

        with Image.open(buffer) as compressed:
            lr_image = compressed.convert("RGB").copy()

        buffer.close()

    return lr_image


# ============================================================
# 2. ПІДГОТОВКА TRAIN / VALID
# ============================================================

def process_split(split_name, source_dir, report):
    """Перевіряє та обробляє одну частину датасету."""

    files = get_image_files(source_dir)

    hr_output = OUTPUT_DIR / split_name / "HR"
    lr_output = OUTPUT_DIR / split_name / "LR"

    hr_output.mkdir(parents=True, exist_ok=True)
    lr_output.mkdir(parents=True, exist_ok=True)

    report.append(f"\n{'=' * 60}")
    report.append(f"Набір: {split_name.upper()}")
    report.append(f"Знайдено файлів: {len(files)}")
    report.append(f"{'=' * 60}")

    successful = 0
    failed = 0

    for index, path in enumerate(files, start=1):

        info = inspect_image(path)

        if not info["valid"]:
            failed += 1
            report.append(
                f"[ПОМИЛКА] {path.name}: {info['error']}"
            )
            continue

        try:
            sizes = prepare_image(
                path,
                hr_output,
                lr_output
            )

            successful += 1

            report.append(
                f"[OK] {path.name} | "
                f"Режим: {info['mode']} | "
                f"Каналів: {info['channels']} | "
                f"Початковий розмір: {sizes['original_size']} | "
                f"HR: {sizes['hr_size']} | "
                f"LR: {sizes['lr_size']}"
            )

        except Exception as error:
            failed += 1
            report.append(
                f"[ПОМИЛКА ОБРОБКИ] {path.name}: {error}"
            )

        if index % 50 == 0:
            print(
                f"{split_name}: оброблено {index}/{len(files)}"
            )

    report.append(f"\nУспішно оброблено: {successful}")
    report.append(f"Помилок: {failed}")

    print(
        f"{split_name}: успішно {successful}, помилок {failed}"
    )

    return successful, failed


def prepare_dataset():
    """Головна функція підготовки DIV2K."""

    report = []

    print("Початок підготовки DIV2K...")
    print(f"Вхідна папка: {RAW_DIR}")
    print(f"Вихідна папка: {OUTPUT_DIR}")

    train_count, train_errors = process_split(
        "train",
        TRAIN_HR_DIR,
        report
    )

    valid_count, valid_errors = process_split(
        "valid",
        VALID_HR_DIR,
        report
    )

    # Перевірка очікуваної кількості файлів
    report.append("\nПІДСУМКИ")
    report.append(f"Train: {train_count} успішних зображень")
    report.append(f"Valid: {valid_count} успішних зображень")
    report.append(f"Загальна кількість: {train_count + valid_count}")
    report.append(
        f"Загальна кількість помилок: {train_errors + valid_errors}"
    )

    if train_count != 800:
        report.append(
            "УВАГА: очікувалося 800 зображень Train."
        )

    if valid_count != 100:
        report.append(
            "УВАГА: очікувалося 100 зображень Valid."
        )

    report.append(
        "\nТестова вибірка не оброблялася: "
        "у вихідній конфігурації вона не задана."
    )

    report_path = OUTPUT_DIR / "dataset_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report_path.write_text(
        "\n".join(report),
        encoding="utf-8"
    )

    print("\nПідготовку завершено.")
    print(f"Звіт збережено: {report_path}")


# ============================================================
# 3. PYTORCH DATASET З ДИНАМІЧНИМ КРОПОМ
# ============================================================

class DIV2KDataset(Dataset):
    """
    Dataset для навчання моделі Super-Resolution.

    Повертає:
        LR: тензор [3, 64, 64]
        HR: тензор [3, 256, 256]

    Патчі створюються динамічно, тому окремо на диск
    вони не зберігаються.
    """

    def __init__(
        self,
        root_dir,
        split="train",
        hr_patch_size=256,
        scale=4,
        patches_per_image=16,
        augment=True,
        degradation=False
    ):

        self.root_dir = Path(root_dir)
        self.split = split
        self.hr_patch_size = hr_patch_size
        self.scale = scale
        self.lr_patch_size = hr_patch_size // scale
        self.patches_per_image = patches_per_image
        self.augment = augment
        self.degradation = degradation

        self.hr_dir = self.root_dir / split / "HR"
        self.lr_dir = self.root_dir / split / "LR"

        if not self.hr_dir.exists() or not self.lr_dir.exists():
            raise FileNotFoundError(
                f"Не знайдено HR/LR папки для набору {split}"
            )

        self.hr_files = sorted(
            file for file in self.hr_dir.iterdir()
            if file.suffix.lower() == ".png"
        )

        self.lr_files = sorted(
            file for file in self.lr_dir.iterdir()
            if file.suffix.lower() == ".png"
        )

        hr_names = {file.name for file in self.hr_files}
        lr_names = {file.name for file in self.lr_files}

        if hr_names != lr_names:
            raise ValueError(
                "Набори HR та LR не відповідають один одному."
            )

        if not self.hr_files:
            raise ValueError("У датасеті немає зображень.")

    def __len__(self):
        return len(self.hr_files) * self.patches_per_image

    def __getitem__(self, index):

        image_index = index // self.patches_per_image

        hr_path = self.hr_files[image_index]
        lr_path = self.lr_dir / hr_path.name

        # Завантаження HR та LR
        with Image.open(hr_path) as image:
            hr = np.array(image.convert("RGB"))

        with Image.open(lr_path) as image:
            lr = np.array(image.convert("RGB"))

        hr_h, hr_w = hr.shape[:2]
        lr_h, lr_w = lr.shape[:2]

        # Перевірка відповідності масштабування
        if (
                hr_h != lr_h * self.scale
                or hr_w != lr_w * self.scale
        ):
            raise ValueError(
                f"Невідповідність HR/LR: {hr_path.name}"
            )

        # Перевірка мінімального розміру
        if (
                hr_h < self.hr_patch_size
                or hr_w < self.hr_patch_size
        ):
            raise ValueError(
                f"Зображення замале для кропу: {hr_path.name}"
            )

        # ========================================================
        # 1. СИНХРОННИЙ КРОП HR ТА LR
        # ========================================================

        max_lr_x = lr_w - self.lr_patch_size
        max_lr_y = lr_h - self.lr_patch_size

        if self.augment:
            lr_x = random.randint(0, max_lr_x)
            lr_y = random.randint(0, max_lr_y)
        else:
            # Центральний патч для Validation
            lr_x = max_lr_x // 2
            lr_y = max_lr_y // 2

        # Координати HR
        x = lr_x * self.scale
        y = lr_y * self.scale

        # Вирізання відповідних фрагментів
        hr_patch = hr[
            y:y + self.hr_patch_size,
            x:x + self.hr_patch_size
        ]

        lr_patch = lr[
            lr_y:lr_y + self.lr_patch_size,
            lr_x:lr_x + self.lr_patch_size
        ]

        # ========================================================
        # 2. ДЕГРАДАЦІЯ LR
        # ========================================================

        if self.degradation:
            # Перетворюємо LR-патч у PIL Image
            lr_pil = Image.fromarray(lr_patch)

            # Випадково застосовуємо деградації
            lr_pil = apply_degradations(lr_pil)

            # Повертаємо результат у NumPy
            lr_patch = np.array(lr_pil)

        # ========================================================
        # 3. ПЕРЕТВОРЕННЯ У TENSOR
        # ========================================================

        hr_patch = torch.from_numpy(
            np.ascontiguousarray(hr_patch)
        ).permute(2, 0, 1).float() / 255.0

        lr_patch = torch.from_numpy(
            np.ascontiguousarray(lr_patch)
        ).permute(2, 0, 1).float() / 255.0

        # ========================================================
        # 4. СИНХРОННА ГЕОМЕТРИЧНА АУГМЕНТАЦІЯ
        # ========================================================

        if self.augment:

            # Горизонтальне відображення (p = 0.5)
            if random.random() < 0.5:
                hr_patch = torch.flip(hr_patch, dims=[2])
                lr_patch = torch.flip(lr_patch, dims=[2])

            # Вертикальне відображення (p = 0.5)
            if random.random() < 0.5:
                hr_patch = torch.flip(hr_patch, dims=[1])
                lr_patch = torch.flip(lr_patch, dims=[1])

            # Поворот на 90, 180 або 270 градусів (p = 0.5)
            if random.random() < 0.5:
                k = random.choice([1, 2, 3])

                hr_patch = torch.rot90(
                    hr_patch, k, dims=[1, 2]
                )

                lr_patch = torch.rot90(
                    lr_patch, k, dims=[1, 2]
                )

        return lr_patch.contiguous(), hr_patch.contiguous()


# ============================================================
# 4. ПРИКЛАД ВИКОРИСТАННЯ DATASET ТА DATALOADER
# ============================================================

def create_dataloaders():
    train_dataset = DIV2KDataset(
        root_dir=OUTPUT_DIR,
        split="train",
        hr_patch_size=HR_PATCH_SIZE,
        scale=SCALE,
        patches_per_image=16,
        augment=True,
        degradation=True
    )

    valid_dataset = DIV2KDataset(
        root_dir=OUTPUT_DIR,
        split="valid",
        hr_patch_size=HR_PATCH_SIZE,
        scale=SCALE,
        patches_per_image=1,
        augment=False,
        degradation=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    # Перевірка форми отриманих тензорів
    lr_batch, hr_batch = next(iter(train_loader))

    print("\nПеревірка DataLoader:")
    print("LR batch:", lr_batch.shape)
    print("HR batch:", hr_batch.shape)

    return train_loader, valid_loader


# ============================================================
# 5. ЗАПУСК
# ============================================================

if __name__ == "__main__":

    prepare_dataset()

    # Після підготовки створюємо DataLoader
    train_loader, valid_loader = create_dataloaders()

    print("\nКрок 1 успішно виконано!")
