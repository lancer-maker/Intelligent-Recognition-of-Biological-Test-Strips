"""
=====================================================================
train_speed_up.py —— 训练加速工具模块
=====================================================================
将训练提速的三类优化封装为可复用工具, 供 new_env_train.py 等训练脚本调用:

  1. 混合精度训练 (AMP, Automatic Mixed Precision)
     - amp_enabled(device):           判断是否启用 AMP (仅 CUDA)
     - make_amp_scaler(device):       创建 GradScaler (非 CUDA 时自动禁用, 透明透传)
     - amp_train_step(model, optimizer, scaler, criterion, imgs, projs, labels):
       完成一次 AMP 训练步: 前向(autocast) -> scale(loss) -> backward
       -> scaler.step(optimizer) -> scaler.update

  2. pin_memory DataLoader
     - make_dataloader(dataset, ..., device, ...): 创建 DataLoader, CUDA 设备时
       自动开启 pin_memory, 加速 CPU -> GPU 数据传输

  3. 降低验证与日志频率
     - should_validate(epoch, val_interval): 判断该 epoch 是否执行验证/早停
     - make_epoch_tqdm(iterable, **kwargs):  创建 tqdm 进度条, 内置
       mininterval=1.0, 减少刷新频率, 避免频繁打印打断 GPU 连续计算

说明:
  - 指标计算 (roc_auc_score 等) 均在 CPU (numpy) 上进行:
    推理时先 .cpu().numpy() 再计算, 不在 GPU 上直接算指标。
=====================================================================
"""
import hashlib
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def amp_enabled(device: torch.device) -> bool:
    """是否启用混合精度 (AMP): 仅 CUDA 设备启用, CPU 设备返回 False。"""
    return device.type == "cuda"


def make_amp_scaler(device: torch.device) -> "torch.amp.GradScaler":
    """创建混合精度梯度缩放器 GradScaler。

    非 CUDA 设备返回 enabled=False 的 scaler (透明 no-op, 数值不变),
    保证同一套训练代码在 CPU/CUDA 上都能安全运行。
    """
    return torch.amp.GradScaler("cuda", enabled=amp_enabled(device))


def amp_train_step(model, optimizer, scaler, criterion, imgs, projs, labels) -> float:
    """AMP 训练步: 等价于普通 train step 的混合精度版本。

    Args:
        model: 双流网络
        optimizer: 优化器
        scaler: make_amp_scaler 返回的 GradScaler
        criterion: 损失函数
        imgs: 已移至 device 的图像批 (B, 3, H, W)
        projs: 已移至 device 的投影批 (B, C, L)
        labels: 已移至 device 的标签批 (B, 1)

    Returns:
        float: 该 batch 的 loss 值 (用于累计 train_loss)
    """
    optimizer.zero_grad()
    with torch.amp.autocast(device_type="cuda", enabled=scaler.is_enabled()):
        logits = model(imgs, projs)
        loss = criterion(logits, labels)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return float(loss.item())


def make_dataloader(dataset, batch_size: int, shuffle: bool, num_workers: int,
                    device: torch.device, **kwargs) -> DataLoader:
    """创建 DataLoader; CUDA 设备时自动开启 pin_memory 加速 H2D 传输。

    num_workers>0 时默认开启 persistent_workers=True: 让 worker 进程跨 epoch
    常驻复用, 避免 Windows spawn 模式下【每个 epoch 都重新 spawn 并重新 exec
    主脚本顶层代码】带来的巨大开销与长训练后 spawn 崩溃风险。
    """
    kwargs.setdefault("pin_memory", device.type == "cuda")
    if num_workers > 0:
        # 仅当调用方未显式覆盖时开启 (Windows 上尤其重要)
        kwargs.setdefault("persistent_workers", True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, **kwargs)


def should_validate(epoch: int, val_interval: int = 1) -> bool:
    """判断该 epoch 是否执行验证与早停判断 (每 val_interval 个 epoch 一次)。

    将 val_interval 设为 2~3 可减少验证频率, 降低开销;
    非验证 epoch 不改变早停计数。
    """
    val_interval = max(1, int(val_interval))
    return epoch % val_interval == 0


def make_epoch_tqdm(iterable, **kwargs):
    """创建 epoch 级 tqdm 进度条, 内置 mininterval=1.0 减少刷新频率。"""
    kwargs.setdefault("mininterval", 1.0)
    return tqdm(iterable, **kwargs)


class ProjectionCache:
    """投影 (行平均 + 可选中央ROI列平均) 计算缓存, 避免每 epoch 重复执行昂贵的 CPU 投影提取。

    缓存的是基于【原始图像】计算的确定性投影 (未施加投影颜色增强),
    以 图像路径 + 参数 为键, 同时支持内存缓存与磁盘缓存 (npy/torch 文件)。

    注意:
      - 缓存结果与"增强后提取"的投影在随机性上有区别: 启用缓存后投影不再随
        投影颜色增强变化 (更稳定); 若训练流程依赖投影随机增强, 请自行对缓存
        投影施加轻量扰动, 或不要启用缓存。
      - **默认不启用 (enabled=False)**, 调用方不显式启用时, 走原有 CPU 计算
        逻辑, 完全不影响原本运行。
      - 返回的 tensor 为克隆副本, 共享安全。
    """

    def __init__(self, cache_dir: str = "", enabled: bool = True):
        self.enabled = bool(enabled)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._mem: dict = {}
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _key(image_path: str, use_col_avg: bool, proj_length: int,
             col_roi_fraction: float, col_length: int) -> str:
        return (f"{Path(image_path).resolve()}|col={int(bool(use_col_avg))}"
                f"|pl={proj_length}|rf={col_roi_fraction}|cl={col_length}")

    def _disk_path(self, key: str) -> Path:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{hashlib.md5(key.encode('utf-8')).hexdigest()}.pt"

    def _compute(self, image_path: str, use_col_avg: bool, proj_length: int,
                 col_roi_fraction: float, col_length: int) -> torch.Tensor:
        """计算原始图像的投影 (行平均 + 可选列平均), 返回 (1, L) 或 (2, L)。"""
        import cv2
        from src.data.utils import (
            extract_1d_projection_resampled,
            extract_col_avg_projection_resampled,
        )
        raw_bgr = cv2.imread(str(image_path))
        if raw_bgr is None:
            raise FileNotFoundError(f"无法读取图像文件: {image_path}")
        raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
        row_t = extract_1d_projection_resampled(raw_rgb, target_length=proj_length)
        if use_col_avg:
            col_t = extract_col_avg_projection_resampled(
                raw_rgb, target_length=col_length, roi_fraction=col_roi_fraction
            )
            return torch.cat([row_t, col_t], dim=0)
        return row_t

    def get_or_compute(self, image_path: str, use_col_avg: bool = False,
                       proj_length: int = 512, col_roi_fraction: float = 0.33,
                       col_length: int = 512) -> torch.Tensor:
        """获取 (或计算并缓存) 某图像的投影张量。

        未启用缓存 (enabled=False) 时直接计算返回, 不读写缓存。
        """
        if not self.enabled:
            return self._compute(image_path, use_col_avg, proj_length,
                                 col_roi_fraction, col_length)
        key = self._key(image_path, use_col_avg, proj_length,
                        col_roi_fraction, col_length)
        # 1) 内存缓存
        cached = self._mem.get(key)
        if cached is not None:
            return cached.clone()
        # 2) 磁盘缓存
        disk = self._disk_path(key)
        if disk is not None and disk.exists():
            cached = torch.load(disk, map_location="cpu", weights_only=True)
            self._mem[key] = cached
            return cached.clone()
        # 3) 计算并缓存
        cached = self._compute(image_path, use_col_avg, proj_length,
                               col_roi_fraction, col_length)
        self._mem[key] = cached
        if disk is not None:
            torch.save(cached, str(disk))
        return cached.clone()
