import sys
from pathlib import Path
from typing import List, Tuple, Optional, Callable

import cv2
import torch
import yaml
from torch.utils.data import Dataset
from configs.load_config import get_main_config

try:
    from .utils import extract_1d_projection_resampled, resize_to_standard_dpi
    from .transforms import get_val_transforms, get_projection_transforms
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from data.utils import extract_1d_projection_resampled, resize_to_standard_dpi
    from data.transforms import get_val_transforms, get_projection_transforms



dataset_config = get_main_config()
config_dpi_size = tuple(dataset_config.get("data", {}).get("standard_dpi_size", [220, 505]))
config_proj_length = int(dataset_config.get("data", {}).get("proj_length", 512))


class TestStripDataset(Dataset):
    """
    物理感知型单试纸数据集类
    """
    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transforms: Optional[Callable] = None,
        projection_transforms: Optional[Callable] = None,
        standard_dpi_size: Optional[Tuple[int, int]] = None,  # 标准物理 DPI 像素尺寸
        proj_length: Optional[int] = None                     # 1D 物理重采样点数
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transforms = transforms if transforms is not None else get_val_transforms()
        self.projection_transforms = projection_transforms if projection_transforms is not None else get_projection_transforms()
        self.standard_dpi_size = tuple(standard_dpi_size) if standard_dpi_size is not None else config_dpi_size
        self.proj_length = proj_length if proj_length is not None else config_proj_length

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # 1. 读取绝对原始像素的图像 (不做任何 Resize)
        raw_bgr = cv2.imread(image_path)
        if raw_bgr is None:
            raise FileNotFoundError(f"无法读取文件: {image_path}")
        raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)

        # 2. 【1D流】先对原始图像复制一份施加同样的颜色增强，再基于增强后的图像提取投影，并线性重采样到固定物理采样点 512
        proj_image = raw_rgb.copy()
        if self.projection_transforms is not None:
            proj_image = self.projection_transforms(image=proj_image)['image']
        proj_tensor = extract_1d_projection_resampled(proj_image, target_length=self.proj_length)

        # 3. 【2D流】将图像缩放到“标准 DPI 尺寸”(220, 505)，保证卷积核感受野物理一致
        dpi_aligned_rgb = resize_to_standard_dpi(raw_rgb, target_size=self.standard_dpi_size)

        # 4. 2D 图像增强与 ImageNet 归一化
        augmented = self.transforms(image=dpi_aligned_rgb)
        image_tensor = augmented['image'] # (3, 505, 220)

        label_tensor = torch.tensor(label, dtype=torch.float32)

        return image_tensor, proj_tensor, label_tensor

    # 快速测试代码范例
if __name__ == "__main__":
    try:
        from .transforms import get_train_transforms
    except ImportError:
        from data.transforms import get_train_transforms

    # 模拟数据
    test_paths = ["D:\\software\\vscode\\TOXY\\data\\raw\\P-24\\P0_01_1.jpg.png"]
    test_labels = [1]
    
    dataset = TestStripDataset(
        image_paths=test_paths, 
        labels=test_labels, 
        transforms=get_train_transforms()
    )
    
    img, proj, label = dataset[0]
    
    print("Image Tensor Shape:", img.shape)   # 应输出: torch.Size([3, 505, 220])
    print("Proj Tensor Shape: ", proj.shape)  # 应输出: torch.Size([1, 512])
    print("Label Tensor:       ", label)       # 应输出: tensor(1.)