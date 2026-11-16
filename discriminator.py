
import torch
import torch.nn as nn


# ============================================================
# НАЛАШТУВАННЯ ДИСКРИМІНАТОРА
# ============================================================

IN_CHANNELS = 3

BASE_CHANNELS = 64

NUM_BLOCKS = 6

LEAKY_RELU_SLOPE = 0.2


# ============================================================
# 1. ЗГОРТКОВИЙ БЛОК
# ============================================================

class DiscriminatorBlock(nn.Module):
    """
    Один згортковий блок дискримінатора.

    Зменшує ширину та висоту зображення вдвічі.
    Збільшує кількість каналів ознак.
    """

    def __init__(
        self,
        in_channels,
        out_channels
    ):
        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            nn.LeakyReLU(
                negative_slope=LEAKY_RELU_SLOPE,
                inplace=True
            )
        )

    def forward(self, x):
        return self.block(x)


# ============================================================
# 2. VGG-LIKE DISCRIMINATOR
# ============================================================

class Discriminator(nn.Module):
    """
    Згортковий дискримінатор для Super-Resolution GAN.

    Вхід:
        [B, 3, 256, 256]

    Вихід:
        [B, 1]

    Вихід — логіт, а не ймовірність.
    Sigmoid не використовується.
    """

    def __init__(
        self,
        in_channels=IN_CHANNELS,
        base_channels=BASE_CHANNELS
    ):
        super().__init__()

        # Кількість каналів у шести блоках
        channels = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 8,
            base_channels * 8
        ]

        # ----------------------------------------------------
        # 1. Послідовність згорткових блоків
        # ----------------------------------------------------

        blocks = []

        current_channels = in_channels

        for out_channels in channels:

            blocks.append(
                DiscriminatorBlock(
                    in_channels=current_channels,
                    out_channels=out_channels
                )
            )

            current_channels = out_channels

        self.features = nn.Sequential(*blocks)

        # ----------------------------------------------------
        # 2. Глобальне усереднення ознак
        # ----------------------------------------------------

        self.global_pool = nn.AdaptiveAvgPool2d(
            output_size=1
        )

        # ----------------------------------------------------
        # 3. Класифікаційна частина
        # ----------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                channels[-1],
                256
            ),

            nn.LeakyReLU(
                negative_slope=LEAKY_RELU_SLOPE,
                inplace=True
            ),

            nn.Linear(
                256,
                1
            )
        )

    def forward(self, x):

        # Витягування ознак
        features = self.features(x)

        # Усереднення просторових ознак
        pooled = self.global_pool(features)

        # Отримання одного логіта
        output = self.classifier(pooled)

        return output


# ============================================================
# 3. ПЕРЕВІРКА ДИСКРИМІНАТОРА
# ============================================================

if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Пристрій:", device)

    # Створення дискримінатора
    discriminator = Discriminator().to(device)

    discriminator.eval()

    # Тестові зображення
    test_images = torch.rand(
        2,
        3,
        256,
        256,
        device=device
    )

    # Прямий прохід
    with torch.no_grad():

        logits = discriminator(test_images)

    # Кількість параметрів
    total_params = sum(
        parameter.numel()
        for parameter in discriminator.parameters()
    )

    print("\nПеревірка Discriminator")
    print("--------------------------------")

    print("Вхідний розмір:", test_images.shape)
    print("Вихідний розмір:", logits.shape)

    print("Кількість блоків:", NUM_BLOCKS)

    print(
        "Кількість параметрів:",
        f"{total_params:,}"
    )

    print("Мінімальний логіт:", logits.min().item())
    print("Максимальний логіт:", logits.max().item())

    # Перевірка форми
    assert logits.shape == (2, 1), (
        f"Неправильна форма виходу: {logits.shape}"
    )

    # Перевірка числової коректності
    assert torch.isfinite(logits).all(), (
        "Вихід містить NaN або Inf!"
    )

    print("\nФорма виходу правильна.")
    print("Некоректних числових значень не виявлено.")

    print("\nДискримінатор успішно створений!")
