
import torch
import torch.nn as nn


# ============================================================
# НАЛАШТУВАННЯ ГЕНЕРАТОРА
# ============================================================

IN_CHANNELS = 3
OUT_CHANNELS = 3

FEATURE_CHANNELS = 64
GROWTH_CHANNELS = 32

NUM_RRDB_BLOCKS = 8
SCALE_FACTOR = 4

RESIDUAL_SCALE = 0.2


# ============================================================
# 1. RESIDUAL DENSE BLOCK
# ============================================================

class ResidualDenseBlock(nn.Module):
    """
    Щільний залишковий блок.

    Кожен наступний згортковий шар отримує ознаки
    попередніх шарів через конкатенацію.
    """

    def __init__(self, channels=64, growth_channels=32):
        super().__init__()

        self.conv1 = nn.Conv2d(
            channels,
            growth_channels,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            channels + growth_channels,
            growth_channels,
            kernel_size=3,
            padding=1
        )

        self.conv3 = nn.Conv2d(
            channels + growth_channels * 2,
            growth_channels,
            kernel_size=3,
            padding=1
        )

        self.conv4 = nn.Conv2d(
            channels + growth_channels * 3,
            growth_channels,
            kernel_size=3,
            padding=1
        )

        self.conv5 = nn.Conv2d(
            channels + growth_channels * 4,
            channels,
            kernel_size=3,
            padding=1
        )

        self.activation = nn.LeakyReLU(
            negative_slope=0.2,
            inplace=True
        )

    def forward(self, x):

        x1 = self.activation(self.conv1(x))

        x2 = self.activation(
            self.conv2(torch.cat((x, x1), dim=1))
        )

        x3 = self.activation(
            self.conv3(torch.cat((x, x1, x2), dim=1))
        )

        x4 = self.activation(
            self.conv4(torch.cat((x, x1, x2, x3), dim=1))
        )

        x5 = self.conv5(
            torch.cat((x, x1, x2, x3, x4), dim=1)
        )

        # Залишкове з'єднання
        return x + RESIDUAL_SCALE * x5


# ============================================================
# 2. RESIDUAL-IN-RESIDUAL DENSE BLOCK
# ============================================================

class RRDB(nn.Module):
    """
    RRDB складається з трьох Residual Dense Blocks.

    Вихід кожного блоку передається наступному.
    Наприкінці додається зовнішнє залишкове з'єднання.
    """

    def __init__(self, channels=64, growth_channels=32):
        super().__init__()

        self.rdb1 = ResidualDenseBlock(
            channels,
            growth_channels
        )

        self.rdb2 = ResidualDenseBlock(
            channels,
            growth_channels
        )

        self.rdb3 = ResidualDenseBlock(
            channels,
            growth_channels
        )

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)

        return x + RESIDUAL_SCALE * out


# ============================================================
# 3. PIXELSHUFFLE UPSAMPLING
# ============================================================

class PixelShuffleUpsample(nn.Module):
    """
    Збільшує просторову роздільну здатність у 2 рази.

    Conv2d формує 4 * channels карт ознак,
    PixelShuffle перетворює їх на зображення
    з удвічі більшою шириною та висотою.
    """

    def __init__(self, channels=64):
        super().__init__()

        self.upsample = nn.Sequential(

            nn.Conv2d(
                channels,
                channels * 4,
                kernel_size=3,
                padding=1
            ),

            nn.PixelShuffle(upscale_factor=2),

            nn.LeakyReLU(
                negative_slope=0.2,
                inplace=True
            )
        )

    def forward(self, x):
        return self.upsample(x)


# ============================================================
# 4. RRDB GENERATOR
# ============================================================

class RRDBGenerator(nn.Module):
    """
    Модифікований генератор Super-Resolution.

    Вхід:
        LR-зображення [B, 3, H, W]

    Вихід:
        SR-зображення [B, 3, H*4, W*4]

    Архітектура:
        1. Вхідна згортка.
        2. Послідовність RRDB.
        3. Згортка об'єднання ознак.
        4. PixelShuffle x2.
        5. PixelShuffle x2.
        6. Фінальна згортка RGB.
    """

    def __init__(
        self,
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        features=FEATURE_CHANNELS,
        growth_channels=GROWTH_CHANNELS,
        num_blocks=NUM_RRDB_BLOCKS
    ):
        super().__init__()

        # ----------------------------------------------------
        # 1. Початкова згортка
        # ----------------------------------------------------

        self.conv_first = nn.Conv2d(
            in_channels,
            features,
            kernel_size=3,
            padding=1
        )

        # ----------------------------------------------------
        # 2. Послідовність RRDB-блоків
        # ----------------------------------------------------

        self.rrdb_blocks = nn.Sequential(

            *[
                RRDB(
                    channels=features,
                    growth_channels=growth_channels
                )
                for _ in range(num_blocks)
            ]

        )

        # ----------------------------------------------------
        # 3. Згортка об'єднання ознак
        # ----------------------------------------------------

        self.conv_trunk = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            padding=1
        )

        # ----------------------------------------------------
        # 4. Збільшення роздільної здатності ×4
        # ----------------------------------------------------

        self.upsampling = nn.Sequential(

            PixelShuffleUpsample(features),

            PixelShuffleUpsample(features)

        )

        # ----------------------------------------------------
        # 5. Фінальні згорткові шари
        # ----------------------------------------------------

        self.conv_hr = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            padding=1
        )

        self.conv_last = nn.Conv2d(
            features,
            out_channels,
            kernel_size=3,
            padding=1
        )

        self.activation = nn.LeakyReLU(
            negative_slope=0.2,
            inplace=True
        )

    def forward(self, x):

        # Початкові ознаки
        features = self.conv_first(x)

        # Обробка RRDB
        trunk = self.rrdb_blocks(features)

        # Залишкове з'єднання генератора
        trunk = self.conv_trunk(trunk)

        features = features + trunk

        # Збільшення роздільної здатності ×4
        features = self.upsampling(features)

        # Фінальна обробка
        features = self.activation(
            self.conv_hr(features)
        )

        output = self.conv_last(features)

        return output


# ============================================================
# 5. ПЕРЕВІРКА ГЕНЕРАТОРА
# ============================================================

if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Пристрій:", device)

    # Створення генератора
    generator = RRDBGenerator(
        num_blocks=NUM_RRDB_BLOCKS
    ).to(device)

    # Тестове LR-зображення
    test_input = torch.rand(
        1,
        3,
        64,
        64,
        device=device
    )

    # Перевірка прямого проходу
    generator.eval()

    with torch.no_grad():
        output = generator(test_input)

    # Кількість параметрів
    total_params = sum(
        parameter.numel()
        for parameter in generator.parameters()
    )

    print("\nПеревірка RRDB Generator")
    print("--------------------------------")
    print("Вхідний розмір:", test_input.shape)
    print("Вихідний розмір:", output.shape)
    print("Кількість RRDB:", NUM_RRDB_BLOCKS)
    print("Кількість параметрів:", f"{total_params:,}")
    print("Пристрій:", device)

    # Перевірка очікуваної форми
    expected_shape = (1, 3, 256, 256)

    assert output.shape == expected_shape, (
        f"Неправильний розмір виходу: {output.shape}"
    )

    print("\nГенератор успішно створений!")
