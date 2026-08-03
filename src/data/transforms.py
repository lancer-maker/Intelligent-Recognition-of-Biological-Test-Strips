import os
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 禁用 Albumentations 的联网版本检查 (解决 The read operation timed out 警告)
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"


def get_train_transforms() -> A.Compose:
    """训练数据增强流水线 (适配新版 Albumentations API)"""
    return A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.5),
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
    """验证数据预处理"""
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])