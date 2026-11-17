
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# 1. SSIM LOSS
# ==========================================

class SSIMLoss(nn.Module):
    """
    Structural Similarity Loss.

    Вимірює структурну подібність між HR та SR.
    Значення Loss наближається до 0,
    коли зображення стають більш подібними.
    """

    def __init__(
        self,
        window_size=11,
        sigma=1.5,
        data_range=1.0,
        channels=3
    ):
        super().__init__()

        self.window_size = window_size
        self.data_range = data_range
        self.channels = channels

        # Створення одновимірного гаусового ядра
        coordinates = torch.arange(
            window_size,
            dtype=torch.float32
        )

        coordinates -= (window_size - 1) / 2

        gaussian = torch.exp(
            -(coordinates ** 2) / (2 * sigma ** 2)
        )

        gaussian /= gaussian.sum()

        # Двовимірне гаусове ядро
        window = gaussian.unsqueeze(1) @ gaussian.unsqueeze(0)

        window = window.unsqueeze(0).unsqueeze(0)

        # Окреме ядро для кожного каналу RGB
        window = window.expand(
            channels, 1, window_size, window_size
        ).contiguous()

        self.register_buffer("window", window)

    def forward(self, img1, img2):

        if img1.shape != img2.shape:
            raise ValueError(
                f"Розміри зображень не збігаються: "
                f"{img1.shape} та {img2.shape}"
            )

        channels = img1.size(1)

        if channels != self.channels:
            raise ValueError(
                f"Очікується {self.channels} каналів, "
                f"отримано {channels}"
            )

        window = self.window.to(
            device=img1.device,
            dtype=img1.dtype
        )

        padding = self.window_size // 2

        # Середні значення
        mu1 = F.conv2d(
            img1, window, padding=padding, groups=channels
        )

        mu2 = F.conv2d(
            img2, window, padding=padding, groups=channels
        )

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        # Дисперсії та коваріація
        sigma1_sq = (
            F.conv2d(
                img1 * img1,
                window,
                padding=padding,
                groups=channels
            ) - mu1_sq
        )

        sigma2_sq = (
            F.conv2d(
                img2 * img2,
                window,
                padding=padding,
                groups=channels
            ) - mu2_sq
        )

        sigma12 = (
            F.conv2d(
                img1 * img2,
                window,
                padding=padding,
                groups=channels
            ) - mu1_mu2
        )

        # Константи SSIM
        c1 = (0.01 * self.data_range) ** 2
        c2 = (0.03 * self.data_range) ** 2

        # Формула SSIM
        numerator = (
            (2 * mu1_mu2 + c1) *
            (2 * sigma12 + c2)
        )

        denominator = (
            (mu1_sq + mu2_sq + c1) *
            (sigma1_sq + sigma2_sq + c2)
        )

        ssim_map = numerator / denominator

        ssim_value = ssim_map.mean()

        # Перетворення подібності на функцію втрат
        return 1.0 - ssim_value


# ==========================================
# 2. COMBINED TOTAL LOSS
# ==========================================

class CombinedLoss(nn.Module):
    """
    Комбінована функція втрат:

    L_total = lambda1 * L1
            + lambda2 * SSIM
            + lambda3 * Adversarial
    """

    def __init__(
        self,
        lambda_pixel=1.0,
        lambda_ssim=0.1,
        lambda_adv=0.001
    ):
        super().__init__()

        self.lambda_pixel = lambda_pixel
        self.lambda_ssim = lambda_ssim
        self.lambda_adv = lambda_adv

        self.l1_loss = nn.L1Loss()

        self.ssim_loss = SSIMLoss()

    def forward(
        self,
        sr,
        hr,
        adv_loss=None,
        use_adversarial=False
    ):

        # 1. Похибка між пікселями
        pixel_loss = self.l1_loss(sr, hr)

        # 2. Структурна похибка
        ssim_loss = self.ssim_loss(
            torch.clamp(sr, 0.0, 1.0),
            hr
        )

        # 3. Adversarial Loss
        if use_adversarial:
            if adv_loss is None:
                raise ValueError(
                    "Для GAN-етапу необхідно передати adv_loss"
                )

            adversarial_loss = adv_loss
        else:
            adversarial_loss = sr.new_zeros(())

        # Загальна функція втрат
        total_loss = (
            self.lambda_pixel * pixel_loss
            + self.lambda_ssim * ssim_loss
            + self.lambda_adv * adversarial_loss
        )

        return {
            "total": total_loss,
            "pixel": pixel_loss,
            "ssim": ssim_loss,
            "adversarial": adversarial_loss
        }


# ==========================================
# 3. TESTING
# ==========================================

if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Пристрій:", device)

    # Ініціалізація функцій втрат
    criterion = CombinedLoss().to(device)

    # Створення тестових зображень
    hr = torch.rand(
        2, 3, 256, 256,
        device=device
    )

    sr = torch.rand(
        2, 3, 256, 256,
        device=device,
        requires_grad=True
    )

    # ======================================
    # Тест 1. Попереднє навчання
    # ======================================

    losses = criterion(
        sr,
        hr,
        use_adversarial=False
    )

    print("\nТест 1. Reconstruction Loss")
    print("--------------------------------")

    for name, value in losses.items():
        print(f"{name}: {value.item():.6f}")

    # Перевірка градієнтів
    losses["total"].backward()

    assert sr.grad is not None
    assert torch.isfinite(sr.grad).all()

    print("Градієнти успішно обчислені.")

    # ======================================
    # Тест 2. Перевірка SSIM
    # ======================================

    identical_loss = criterion.ssim_loss(hr, hr)

    print("\nТест 2. SSIM для однакових зображень")
    print("--------------------------------")

    print(
        "SSIM Loss:",
        f"{identical_loss.item():.8f}"
    )

    assert abs(identical_loss.item()) < 1e-5

    print("Перевірка SSIM успішна.")

    # ======================================
    # Тест 3. Перевірка GAN-режиму
    # ======================================

    fake_adv_loss = torch.tensor(
        0.5,
        device=device
    )

    gan_losses = criterion(
        sr.detach(),
        hr,
        adv_loss=fake_adv_loss,
        use_adversarial=True
    )

    print("\nТест 3. GAN Loss")
    print("--------------------------------")

    for name, value in gan_losses.items():
        print(f"{name}: {value.item():.6f}")

    assert torch.isfinite(gan_losses["total"])

    print("\nУсі тести успішно пройдені!")
