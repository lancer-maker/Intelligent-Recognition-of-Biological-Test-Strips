"""逐个导出训练增强后的图像。

请先填写 INPUT_IMAGE 和 OUTPUT_DIR，再运行本脚本。
每个增强都从同一张原图开始执行，输出文件名包含增强名称。
"""

from pathlib import Path

import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2

from src.data.transforms import get_train_transforms


# TODO: 填写待增强的图片路径和输出目录。
INPUT_IMAGE = r"data\raw\P-24\P02_03_1.jpg"
OUTPUT_DIR = r"outputs\augmented images"


def export_augmented_images(input_image: str, output_dir: str) -> None:
    """将训练流水线中的每个图像增强单独保存为一张图片。"""
    input_path = Path(input_image)
    output_path = Path(output_dir)

    if not input_path.is_file():
        raise FileNotFoundError(f"输入图片不存在: {input_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    image_bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"无法读取输入图片: {input_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # 复用项目中的训练增强定义，但单独执行每个可保存为图片的增强。
    transforms = get_train_transforms().transforms
    image_transforms = [
        transform
        for transform in transforms
        if not isinstance(transform, (ToTensorV2,))
        and transform.__class__.__name__ != "Normalize"
    ]

    stem = input_path.stem
    for transform in image_transforms:
        original_probability = transform.p
        transform.p = 1.0
        try:
            result = transform(image=image_rgb.copy())["image"]
        finally:
            transform.p = original_probability

        if not isinstance(result, np.ndarray):
            raise TypeError(
                f"增强 {transform.__class__.__name__} 未返回可保存的图像数组"
            )

        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        transform_name = transform.__class__.__name__
        save_path = output_path / f"{stem}_{transform_name}.png"
        if not cv2.imwrite(str(save_path), result_bgr):
            raise OSError(f"保存图片失败: {save_path}")
        print(f"已保存: {save_path}")


if __name__ == "__main__":
    if not INPUT_IMAGE or not OUTPUT_DIR:
        raise SystemExit("请先在脚本顶部填写 INPUT_IMAGE 和 OUTPUT_DIR。")
    export_augmented_images(INPUT_IMAGE, OUTPUT_DIR)