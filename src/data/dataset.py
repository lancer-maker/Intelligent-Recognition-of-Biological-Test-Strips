import sys
from pathlib import Path
from typing import List, Tuple, Optional, Callable

import cv2
import torch
import yaml
from torch.utils.data import Dataset
from configs.load_config import get_main_config

try:
    from .utils import extract_1d_projection_resampled, extract_col_avg_projection_resampled, resize_to_standard_dpi
    from .transforms import get_val_transforms, get_projection_transforms
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from data.utils import extract_1d_projection_resampled, extract_col_avg_projection_resampled, resize_to_standard_dpi
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
        projection_transforms: Optional[Callable] = None,   # None=默认随机增强; 传 False 可关闭(确定性)
        standard_dpi_size: Optional[Tuple[int, int]] = None,  # 标准物理 DPI 像素尺寸
        proj_length: Optional[int] = None,                    # 1D 物理重采样点数
        use_col_avg: Optional[bool] = None,                   # 是否启用中央ROI列平均(双通道)
        col_roi_fraction: Optional[float] = None,             # 列平均中央ROI比例
        col_length: Optional[int] = None,                     # 列平均横向重采样长度
        proj_cache: Optional[object] = None,                  # 投影缓存 (train_speed_up.ProjectionCache), 默认 None 不启用
        skip_projection: bool = False                         # True=跳过 1D 投影提取(仅图像流/去投影消融)
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transforms = transforms if transforms is not None else get_val_transforms()
        # None -> 默认投影随机增强; False -> 显式关闭增强 (结果确定可复现)
        self.projection_transforms = get_projection_transforms() if projection_transforms is None else projection_transforms
        self.standard_dpi_size = tuple(standard_dpi_size) if standard_dpi_size is not None else config_dpi_size
        self.proj_length = proj_length if proj_length is not None else config_proj_length
        self.proj_cache = proj_cache                          # None 时走原有投影计算逻辑
        self.skip_projection = bool(skip_projection)          # True 时不提取 1D 投影 (占位零张量)

        # 双通道 (行平均 + 列平均) 配置, 默认从主配置读取
        _data_cfg = dataset_config.get("data", {})
        self.use_col_avg = bool(_data_cfg.get("proj_use_col_avg", False)) if use_col_avg is None else bool(use_col_avg)
        self.col_roi_fraction = float(_data_cfg.get("proj_col_roi_fraction", 0.33)) if col_roi_fraction is None else float(col_roi_fraction)
        self.col_length = int(_data_cfg.get("proj_col_length", 512)) if col_length is None else int(col_length)

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

        # 2. 【1D流】提取投影 (行平均 + 可选列平均), 并线性重采样到固定物理采样点 512
        #    skip_projection=True (纯图像流/去投影消融): 不提取投影, 用占位零张量保持批次形状
        if self.skip_projection:
            proj_tensor = torch.zeros((1, self.proj_length), dtype=torch.float32)
        elif self.proj_cache is not None:
            # 使用投影缓存: 基于原始图像计算, 跳过投影颜色增强 (加速数据加载)
            proj_tensor = self.proj_cache.get_or_compute(
                image_path,
                use_col_avg=self.use_col_avg,
                proj_length=self.proj_length,
                col_roi_fraction=self.col_roi_fraction,
                col_length=self.col_length,
            )
        else:
            # 原有逻辑: 对原始图像副本施加投影颜色增强后再提取投影
            proj_image = raw_rgb.copy()
            if self.projection_transforms:               # 传 False 时跳过投影增强 -> 确定性
                proj_image = self.projection_transforms(image=proj_image)['image']
            proj_tensor = extract_1d_projection_resampled(proj_image, target_length=self.proj_length)
            # 可选: 叠加中央ROI列平均(横向)投影 -> 双通道 (2, proj_length); 两个向量经过相同预处理
            if self.use_col_avg:
                col_tensor = extract_col_avg_projection_resampled(
                    proj_image, target_length=self.col_length, roi_fraction=self.col_roi_fraction
                )
                proj_tensor = torch.cat([proj_tensor, col_tensor], dim=0)  # (2, 512)

        # 注: 物理坐标位置编码不再在数据集层拼接(此前为固定坐标通道);
        #     现由 src/models/proj_stream.py 内部实现 (可学习位置通道 + 固定坐标拼到输出)。

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