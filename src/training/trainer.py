# 封装单次折叠（Single Fold）的两阶段训练循环、验证以及早停机制
import os
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.load_config import get_main_config
from .train_utils import get_weighted_bce_loss, get_optimizer

config = get_main_config().get("training", {})
stage1_config = config.get("stage1", {})
stage2_config = config.get("stage2", {})


class StageTrainer:
    """
    两阶段模型训练器。
    负责阶段一（冻结2D）与阶段二（解冻微调）的训练调度、早停和最佳模型保存。
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        save_dir: str,
        pos_weight: float = 1.0             # 正样本权重在实际训练中于cross_val.py中动态计算
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.criterion = get_weighted_bce_loss(pos_weight, device)

    def fit(
        self, 
        fold_info: str = "",                  # 定义loocv折叠编号
        phase1_epochs: Optional[int] = None,
        phase2_epochs: Optional[int] = None,
        patience: Optional[int] = None,
        phase1_lr: Optional[float] = None,
        phase2_lr: Optional[float] = None,
        weight_decay: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        执行两阶段完整训练。
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (最佳验证集预测概率, 验证集真实标签)
        """
        phase1_epochs = phase1_epochs if phase1_epochs is not None else int(stage1_config.get("epochs", 20))
        phase2_epochs = phase2_epochs if phase2_epochs is not None else int(stage2_config.get("epochs", 10))
        patience = patience if patience is not None else int(config.get("early_stopping_patience", 5))
        phase1_lr = phase1_lr if phase1_lr is not None else float(stage1_config.get("learning_rate", 1e-4))
        phase2_lr = phase2_lr if phase2_lr is not None else float(stage2_config.get("learning_rate", 1e-5))
        weight_decay = weight_decay if weight_decay is not None else float(config.get("weight_decay", 1e-3))

        # ==================== 阶段一：冻结 2D 图像流 ====================
        print(">>> Stage 1: Freezing 2D Stream (Training 1D CNN & Head) <<<")
        self.model.freeze_image_stream()
        optimizer1 = get_optimizer(self.model, lr=phase1_lr, weight_decay=weight_decay)
        self._run_epochs(
            optimizer1,
            epochs=phase1_epochs,
            patience=patience,
            stage_name="Stage1",
            fold_info=fold_info,
        )

        # ==================== 阶段二：解冻全网络微调 ====================
        print(">>> Stage 2: Unfreezing All Streams (Fine-tuning) <<<")
        self.model.unfreeze_image_stream()
        optimizer2 = get_optimizer(self.model, lr=phase2_lr, weight_decay=weight_decay) # 降低学习率
        best_preds, best_labels = self._run_epochs(
            optimizer2,
            epochs=phase2_epochs,
            patience=patience,
            stage_name="Stage2",
            fold_info=fold_info,
        )

        return best_preds, best_labels

    def _run_epochs(
        self, 
        optimizer: torch.optim.Optimizer, 
        epochs: int, 
        patience: int, 
        stage_name: str,
        fold_info: str = ""
    ) -> Tuple[np.ndarray, np.ndarray]:
        """训练循环内部执行逻辑"""
        best_val_loss = float('inf')
        best_preds, best_labels = None, None
        patience_counter = 0

    # ------------------------------------------------
        # 格式化前缀标题，例如 "[1/24] Stage1"
        prefix = f"[{fold_info}] " if fold_info else ""
        desc_title = f"{prefix}{stage_name}"

        # 主 Epoch 实时进度条
        epoch_pbar = tqdm(
            range(1, epochs + 1), 
            desc=desc_title, 
            leave=True, 
            dynamic_ncols=True
        )# ------------------------------------------------

        for epoch in epoch_pbar:
            # 1. 训练一轮
            self.model.train()
            train_loss = 0.0
        # ------------------------------------------------
            batch_pbar = tqdm(
                self.train_loader, 
                desc=f"  └─ Epoch {epoch}/{epochs}", 
                leave=False, 
                dynamic_ncols=True
            )# ------------------------------------------------

            for imgs, projs, labels in batch_pbar:
                imgs = imgs.to(self.device)
                projs = projs.to(self.device)
                labels = labels.to(self.device).unsqueeze(1) # (B, 1)

                optimizer.zero_grad()
                logits = self.model(imgs, projs)
                loss = self.criterion(logits, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
            # ------------------------------------------------
            # 实时刷新当前 Batch 的 Loss
                batch_pbar.set_postfix({"b_loss": f"{loss.item():.4f}"})
                # ------------------------------------------------
            train_loss /= len(self.train_loader)

            # 2. 验证一轮
            val_loss, preds, labels = self._validate()

            # ------------------------------------------------
            # 实时更新外层 Epoch 进度条的尾部状态 (显示各 Loss 和 早停计数)
            epoch_pbar.set_postfix({
                "t_loss": f"{train_loss:.4f}",
                "v_loss": f"{val_loss:.4f}",
                "best_v_loss": f"{best_val_loss:.4f}" if best_val_loss != float('inf') else "N/A",
                "patience": f"{patience_counter}/{patience}"
            })# ------------------------------------------------

            # 3. 早停判断与模型保存
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_preds, best_labels = preds, labels
                patience_counter = 0
                # 保存权重
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, "best_model.pth"))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    # ------------------------------------------------------
                    # 使用 tqdm.write 输出早停日志，避免砸乱进度条
                    tqdm.write(f"  ⚠️ {prefix}{stage_name} 触发早停 (Epoch {epoch}/{epochs})")
                    # ------------------------------------------------------
                    break

        # 恢复最佳权重
        best_ckpt = os.path.join(self.save_dir, "best_model.pth")
        if os.path.exists(best_ckpt):
            self.model.load_state_dict(torch.load(best_ckpt, weights_only=True))
            
        return best_preds, best_labels

    def _validate(self) -> Tuple[float, np.ndarray, np.ndarray]:
        """验证函数"""
        self.model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for imgs, projs, labels in self.val_loader:
                imgs = imgs.to(self.device)
                projs = projs.to(self.device)
                labels = labels.to(self.device).unsqueeze(1)

                logits = self.model(imgs, projs)
                loss = self.criterion(logits, labels)
                val_loss += loss.item()

                probs = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(probs)
                all_labels.extend(labels.cpu().numpy())

        val_loss /= len(self.val_loader)
        return val_loss, np.array(all_preds).flatten(), np.array(all_labels).flatten()
