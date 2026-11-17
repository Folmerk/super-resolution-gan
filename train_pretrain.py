
import csv
from pathlib import Path

import torch
from torch.amp import autocast, GradScaler

from rrdb_generator import RRDBGenerator
from prepare_div2k import create_dataloaders
from losses import CombinedLoss


# ==========================================
# 1. НАЛАШТУВАННЯ
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = PROJECT_DIR / "weights"
RESULTS_DIR = PROJECT_DIR / "results"

WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = 100
LEARNING_RATE = 1e-4
BETA1 = 0.9
BETA2 = 0.99

LAMBDA_PIXEL = 1.0
LAMBDA_SSIM = 0.1

# Спочатку запускаємо короткий тест.
# Після успішної перевірки змінити на False.
SMOKE_TEST = True
ACTIVE_EPOCHS = 1 if SMOKE_TEST else EPOCHS
# Кількість пакетів для короткого тесту.
MAX_TRAIN_BATCHES = 10
MAX_VAL_BATCHES = 10

# Для повного навчання використовуються всі пакети.
if not SMOKE_TEST:
    MAX_TRAIN_BATCHES = None
    MAX_VAL_BATCHES = None

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

USE_AMP = DEVICE.type == "cuda"

if SMOKE_TEST:
    CHECKPOINT_PATH = WEIGHTS_DIR / "generator_pretrained_smoke.pth"
    LOG_PATH = RESULTS_DIR / "pretrain_smoke.csv"
else:
    CHECKPOINT_PATH = WEIGHTS_DIR / "generator_pretrained.pth"
    LOG_PATH = RESULTS_DIR / "pretrain_history.csv"


# ==========================================
# 2. ДОПОМІЖНІ ФУНКЦІЇ
# ==========================================

def save_checkpoint(
    path,
    generator,
    optimizer,
    epoch,
    val_loss
):
    checkpoint = {
        "epoch": epoch,
        "generator_state_dict": generator.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss
    }

    torch.save(checkpoint, path)


def initialize_csv(path):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow([
            "epoch",
            "train_total",
            "train_pixel",
            "train_ssim",
            "val_total",
            "val_pixel",
            "val_ssim"
        ])


def append_csv(path, values):
    with open(path, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(values)


# ==========================================
# 3. НАВЧАННЯ ОДНІЄЇ ЕПОХИ
# ==========================================

def train_one_epoch(
    generator,
    train_loader,
    criterion,
    optimizer,
    scaler,
    epoch
):
    generator.train()

    total_loss_sum = 0.0
    pixel_loss_sum = 0.0
    ssim_loss_sum = 0.0

    batches = 0

    for batch_index, (lr, hr) in enumerate(train_loader):

        if (
            MAX_TRAIN_BATCHES is not None
            and batch_index >= MAX_TRAIN_BATCHES
        ):
            break

        lr = lr.to(DEVICE, non_blocking=True)
        hr = hr.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Forward propagation
        with autocast(
            device_type=DEVICE.type,
            enabled=USE_AMP
        ):
            sr = generator(lr)

            losses = criterion(
                sr,
                hr,
                use_adversarial=False
            )

            loss = losses["total"]

        # Backpropagation
        scaler.scale(loss).backward()

        # Оновлення ваг
        scaler.step(optimizer)
        scaler.update()

        total_loss_sum += losses["total"].item()
        pixel_loss_sum += losses["pixel"].item()
        ssim_loss_sum += losses["ssim"].item()

        batches += 1

        if (batch_index + 1) % 100 == 0:
            print(
                f"Epoch [{epoch}/{EPOCHS}] "
                f"Batch [{batch_index + 1}] "
                f"Loss: {loss.item():.6f}"
            )

    if batches == 0:
        raise RuntimeError("Навчальний DataLoader не повернув пакетів.")

    return {
        "total": total_loss_sum / batches,
        "pixel": pixel_loss_sum / batches,
        "ssim": ssim_loss_sum / batches
    }


# ==========================================
# 4. ВАЛІДАЦІЯ
# ==========================================

def validate(
    generator,
    val_loader,
    criterion
):
    generator.eval()

    total_loss_sum = 0.0
    pixel_loss_sum = 0.0
    ssim_loss_sum = 0.0

    batches = 0

    with torch.no_grad():

        for batch_index, (lr, hr) in enumerate(val_loader):

            if (
                MAX_VAL_BATCHES is not None
                and batch_index >= MAX_VAL_BATCHES
            ):
                break

            lr = lr.to(DEVICE, non_blocking=True)
            hr = hr.to(DEVICE, non_blocking=True)

            with autocast(
                device_type=DEVICE.type,
                enabled=USE_AMP
            ):
                sr = generator(lr)

                losses = criterion(
                    sr,
                    hr,
                    use_adversarial=False
                )

            total_loss_sum += losses["total"].item()
            pixel_loss_sum += losses["pixel"].item()
            ssim_loss_sum += losses["ssim"].item()

            batches += 1

    if batches == 0:
        raise RuntimeError("Валідаційний DataLoader не повернув пакетів.")

    return {
        "total": total_loss_sum / batches,
        "pixel": pixel_loss_sum / batches,
        "ssim": ssim_loss_sum / batches
    }


# ==========================================
# 5. ГОЛОВНИЙ ЦИКЛ НАВЧАННЯ
# ==========================================

def main():

    print("=" * 55)
    print("ПОПЕРЕДНЄ НАВЧАННЯ RRDB GENERATOR")
    print("=" * 55)

    print("Пристрій:", DEVICE)
    print("Mixed Precision:", USE_AMP)
    print("Кількість епох:", ACTIVE_EPOCHS)
    print("Режим тестування:", SMOKE_TEST)

    # Завантаження датасету
    train_loader, val_loader = create_dataloaders()

    print("\nTrain batches:", len(train_loader))
    print("Validation batches:", len(val_loader))

    # Ініціалізація генератора
    generator = RRDBGenerator().to(DEVICE)

    # Функція втрат
    criterion = CombinedLoss(
        lambda_pixel=LAMBDA_PIXEL,
        lambda_ssim=LAMBDA_SSIM,
        lambda_adv=0.0
    ).to(DEVICE)

    # Оптимізатор
    optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=LEARNING_RATE,
        betas=(BETA1, BETA2)
    )

    # Scaler для Mixed Precision
    scaler = GradScaler(
        "cuda",
        enabled=USE_AMP
    )

    print(
        "\nПараметрів генератора:",
        f"{sum(p.numel() for p in generator.parameters()):,}"
    )

    initialize_csv(LOG_PATH)

    best_val_loss = float("inf")

    # ======================================
    # Цикл епох
    # ======================================

    for epoch in range(1, ACTIVE_EPOCHS + 1):

        print("\n" + "-" * 55)
        print(f"ЕПОХА {epoch}/{EPOCHS}")
        print("-" * 55)

        train_metrics = train_one_epoch(
            generator,
            train_loader,
            criterion,
            optimizer,
            scaler,
            epoch
        )

        val_metrics = validate(
            generator,
            val_loader,
            criterion
        )

        print("\nРезультати навчання:")
        print(f"Train Total: {train_metrics['total']:.6f}")
        print(f"Train L1:    {train_metrics['pixel']:.6f}")
        print(f"Train SSIM:  {train_metrics['ssim']:.6f}")

        print("\nРезультати валідації:")
        print(f"Val Total:   {val_metrics['total']:.6f}")
        print(f"Val L1:      {val_metrics['pixel']:.6f}")
        print(f"Val SSIM:    {val_metrics['ssim']:.6f}")

        append_csv(
            LOG_PATH,
            [
                epoch,
                train_metrics["total"],
                train_metrics["pixel"],
                train_metrics["ssim"],
                val_metrics["total"],
                val_metrics["pixel"],
                val_metrics["ssim"]
            ]
        )

        # Зберігаємо найкращу модель за валідаційним Loss
        if val_metrics["total"] < best_val_loss:

            best_val_loss = val_metrics["total"]

            save_checkpoint(
                CHECKPOINT_PATH,
                generator,
                optimizer,
                epoch,
                best_val_loss
            )

            print("\nЗбережено новий найкращий checkpoint!")

        print(f"Найкращий Val Loss: {best_val_loss:.6f}")

    print("\n" + "=" * 55)
    print("НАВЧАННЯ ЗАВЕРШЕНО")
    print("=" * 55)

    print("Checkpoint:", CHECKPOINT_PATH)
    print("CSV журнал:", LOG_PATH)
    print(f"Найкращий Val Loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
