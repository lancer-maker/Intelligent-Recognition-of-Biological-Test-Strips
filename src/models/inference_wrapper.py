# 推理封装模块，用于模型导出（ONNX）与工程部署预测
import torch
import torch.nn as nn


class InferenceWrapper(nn.Module):
    """
    推理专用封装包装类。
    对 DualStreamStripNet 的 Logits 输出自动施加 Sigmoid，输出概率 [0, 1]。
    非常适合用于脚本预测与导出 ONNX 部署模型。
    """
    def __init__(self, model: nn.Module):
        super(InferenceWrapper, self).__init__()
        self.model = model
        self.model.eval() # 强制设为评估模式 (关闭 Dropout/BN 动态更新)

    def forward(self, img_tensor: torch.Tensor, proj_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img_tensor: (B, 3, 505, 220) 2D 图像
            proj_tensor: (B, 1, 512) 1D 物理投影
            
        Returns:
            probs: 阳性概率 Tensor (B, 1)，取值范围 [0.0, 1.0]
        """
        logits = self.model(img_tensor, proj_tensor)
        probs = torch.sigmoid(logits)
        return probs