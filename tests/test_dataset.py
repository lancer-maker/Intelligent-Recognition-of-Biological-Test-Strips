import os

# 必须在 import albumentations 之前设置, 否则其联网版本检查会在导入时触发
# (产生无害但扰乱输出/退出码的 UserWarning)
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.data.dataset import TestStripDataset
from src.data.utils import extract_1d_projection_resampled


def test_dataset_applies_projection_color_augmentation_before_projection(tmp_path):
    image_path = tmp_path / "strip.png"
    image = np.zeros((40, 80, 3), dtype=np.uint8)
    image[:, :, 0] = 100
    image[:, :, 1] = 150
    image[:, :, 2] = 200
    cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    def projection_transform(image):
        adjusted = image.copy()
        adjusted[:, :, 0] += 30
        adjusted[:, :, 1] -= 10
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        return {"image": adjusted}

    dataset = TestStripDataset(
        image_paths=[str(image_path)],
        labels=[1],
        transforms=A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]),
        projection_transforms=projection_transform,
        standard_dpi_size=(80, 40),
        proj_length=16,
        use_col_avg=False,             # 测试聚焦"投影颜色增强", 关闭列平均以免受主配置影响
    )

    image_tensor, proj_tensor, label_tensor = dataset[0]

    expected_proj = extract_1d_projection_resampled(
        np.asarray(projection_transform(image)["image"], dtype=np.uint8),
        target_length=16,
    )

    assert label_tensor.item() == 1.0
    assert image_tensor.shape == (3, 40, 80)
    assert proj_tensor.shape == (1, 16)
    assert np.allclose(proj_tensor.numpy(), expected_proj.numpy())
