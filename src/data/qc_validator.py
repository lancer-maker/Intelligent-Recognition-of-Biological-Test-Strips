# 质控试纸判断逻辑
import cv2
import numpy as np
from typing import Tuple, Union
from .utils import read_image_rgb


class QCValidator:
    """
    质控纸带 (QC Strip) 校验器。
    通过提取质控纸带的纵向投影，检测预设区域内是否存在合格的质控峰 (C线)。
    """
    def __init__(self, peak_threshold: float = 0.25, qc_window: Tuple[int, int] = (50, 500)):
        """
        Args:
            peak_threshold: 质控峰相对高度的最小有效阈值 (0 ~ 1.0)
            qc_window: 质控线 (C线) 预计出现的纵向像素索引区间 (start_y, end_y)
        """
        self.peak_threshold = peak_threshold
        self.qc_window = qc_window

    def validate(self, qc_image_input: Union[str, np.ndarray]) -> Tuple[bool, float]:
        """
        校验质控纸带是否有效。
        
        Args:
            qc_image_input: 质控图像的物理路径 (str) 或已读取的 RGB 数组 (np.ndarray)
            
        Returns:
            Tuple[bool, float]: (是否有效, 检测到的波峰相对强度)
        """
        # 1. 统一转为 RGB 图像 (质控纸带标准尺寸 700x220)
        if isinstance(qc_image_input, str):
            img_rgb = read_image_rgb(qc_image_input, target_size=(220, 700))
        else:
            img_rgb = qc_image_input

        # 2. 转为灰度图并反转灰度 (试纸背景为白色~255，条带为暗色波谷；反转后条带变为高亮度波峰)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        inv_gray = 255.0 - gray.astype(np.float32)

        # 3. 沿宽度计算纵向投影
        proj = np.mean(inv_gray, axis=1)

        # 4. 截取预设的 C线 出现窗口
        start_y, end_y = self.qc_window
        window_proj = proj[start_y:end_y]

        if len(window_proj) == 0:
            return False, 0.0

        # 5. 计算信号峰值强度
        max_signal = np.max(window_proj)
        min_baseline = np.min(proj) # 整体背景基线
        
        # 归一化波峰高度
        peak_height = (max_signal - min_baseline) / 255.0

        # 6. 判断是否超过有效阈值
        is_valid = bool(peak_height >= self.peak_threshold)

        return is_valid, float(peak_height)