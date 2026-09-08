import os

# 必须在 import albumentations 之前设置, 否则其联网版本检查会在导入时触发
# (产生无害但扰乱输出/退出码的 UserWarning, 如 "Error fetching version info")
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


class DeterministicCLAHE(A.ImageOnlyTransform):
    """确定性 CLAHE (cv2, LAB 的 L 通道), 替代 A.CLAHE。

    注意: 本项目安装的 albumentations 2.0.8 中 A.CLAHE 即使 p=1 也含随机性
    (同输入两次输出不同), 会导致验证/测试/可解释性结果不可复现。
    本实现改用 cv2.createCLAHE 固定参数, 输入相同时输出完全一致。
    """

    def __init__(self, clip_limit: float = 2.0, tile_grid_size=(8, 8), p: float = 1.0):
        super().__init__(p=p)
        self.clip_limit = float(clip_limit)
        self.tile_grid_size = tuple(int(t) for t in tile_grid_size)

    def apply(self, img, **params):
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit,
                                tileGridSize=self.tile_grid_size)
        l_ch = clahe.apply(l_ch)
        return cv2.cvtColor(cv2.merge((l_ch, a_ch, b_ch)), cv2.COLOR_LAB2RGB)


def get_projection_transforms() -> A.Compose:
    """投影流专用的仅颜色增强流水线。避免几何变换改变投影的物理对应关系。

    (训练阶段专用随机增强; 测试/验证如需确定结果请关闭投影增强)
    """
    return A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.5),
        A.GaussNoise(p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
    ])


def get_train_transforms() -> A.Compose:
    """训练数据增强流水线 (适配新版 Albumentations API)"""
    return A.Compose([
        A.ColorJitter(
            brightness=0.5,
            contrast=0.5,
            saturation=0.5,
            hue=0.2,
            p=0.8,
        ),
        A.RandomGamma(gamma_limit=(70, 130), p=0.7),
        # 新版 GaussNoise 语法
        A.GaussNoise(p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        # 新版 CoarseDropout 语法
        A.CoarseDropout(num_holes_range=(1, 2), hole_height_range=(8, 20), hole_width_range=(8, 40), fill=0, p=0.3),
        # 推荐使用 Affine 替代 ShiftScaleRotate
        A.Affine(scale=(0.95, 1.05), translate_percent=(-0.03, 0.03), rotate=0, p=0.5),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


def get_val_transforms() -> A.Compose:
    """验证/测试数据预处理 (确定性): 局部对比度拉伸(确定性 CLAHE) + 归一化。

    使用 DeterministicCLAHE 替代随机性的 A.CLAHE, 保证同一输入多次处理结果一致,
    使验证/测试/可解释性输出可复现。
    """
    return A.Compose([
        DeterministicCLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])