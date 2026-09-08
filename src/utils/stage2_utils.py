"""
=====================================================================
stage2_utils.py —— Stage2 训练策略功能模块 (余弦退火 + 逐层解冻)
=====================================================================
供 new_env_train.py 的 Stage2 (解冻微调) 阶段使用, 包含两大策略:

1. Stage2 优化器 + 余弦退火 (CosineAnnealingLR)
   - 优化器: AdamW (参数为模型全部参数, 以便渐进解冻时新解冻层能立即参与训练;
     初始 requires_grad=False 的参数不会产生梯度, 不会被更新)
   - 调度器: CosineAnnealingLR, T_max = Stage2 的 epoch 数,
     使学习率从初始值 (lr) 沿余弦曲线逐渐衰减到接近 0 (eta_min, 默认 1e-7)
   - 每个 epoch 训练结束后调用 scheduler.step()

2. 逐层解冻 (Progressive Unfreezing)
   - Stage2 开始前默认冻结所有层 (freeze_all_backbone)
   - 解冻计划完全由配置驱动 (new_env_train 从 main_config.yaml 读取并传入):
       * interval (unfreeze_interval)    : 每多少 epoch 解冻一层
       * start_block (unfreeze_start_block): 从哪个 backbone block 开始解冻
         (EfficientNet-B0: blocks.0~blocks.6, 最深为 6; 一般从最深往前解冻)
       * num_layers (unfreeze_num_layers): 从 start 向前共解冻多少层
   - ProgressiveUnfreezeScheduler 记录每次解冻对应的 (epoch, 层名),
     供终端日志与最终训练报告标注
=====================================================================
"""
from typing import List, Optional, Tuple

import torch


def get_backbone_block_names(model: torch.nn.Module) -> List[str]:
    """返回 EfficientNet backbone 的 block 名列表 (如 ['blocks.0', ..., 'blocks.6'])。

    通过 timm 的 backbone.blocks (nn.Sequential) 长度动态获取, 不写死层数。
    """
    blocks = model.image_stream.backbone.blocks
    n = len(blocks)
    return [f"blocks.{i}" for i in range(n)]


def freeze_all_backbone(model: torch.nn.Module) -> None:
    """冻结 2D 图像流 backbone 的所有层 (Stage2 开始前的默认状态)。"""
    for param in model.image_stream.backbone.parameters():
        param.requires_grad = False


def unfreeze_backbone_block(model: torch.nn.Module, block_name: str) -> None:
    """解冻指定的 backbone block (如 'blocks.6'), 将其参数设为可训练。"""
    module = model.image_stream.backbone
    for part in block_name.split("."):
        module = getattr(module, part)
    for param in module.parameters():
        param.requires_grad = True


def create_stage2_optimizer(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float = 1e-3,
    t_max: int = 15,
    eta_min: float = 1e-7,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """创建 Stage2 优化器 (AdamW) 与余弦退火调度器 (CosineAnnealingLR)。

    Args:
        model: 双流网络 (需已包含可训练/冻结状态; 传入全部参数以支持渐进解冻)
        lr: Stage2 初始学习率
        weight_decay: L2 权重衰减
        t_max: 余弦退火周期 = Stage2 的 epoch 数
        eta_min: 学习率衰减下限 (接近 0)

    Returns:
        (optimizer, scheduler): 每 epoch 结束后调用 scheduler.step()
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=t_max, eta_min=eta_min
    )
    return optimizer, scheduler


class ProgressiveUnfreezeScheduler:
    """逐层解冻调度器: 管理 EfficientNet backbone 的渐进解冻计划。

    用法 (在 _run_epochs 的每个 epoch 训练前调用):
        newly = unfreezer.apply_for_epoch(epoch, stage_name)
        # newly 非空 -> 本次解冻了这些层, 可输出日志/报告标注

    参数 (供配置驱动; None 表示使用默认):
        interval   : 每多少 epoch 解冻一层
        start_block: 解冻起始 block 下标 (None = 从最深一层开始)
        num_layers : 共解冻多少层 (None = 从起始层一直解冻到 blocks.0)

    属性:
        unfreezed: 已解冻层名列表 (累计, 保持解冻顺序)
        unfreeze_log: [(epoch, block_name), ...] 解冻记录, 用于最终报告
    """

    def __init__(
        self,
        model: torch.nn.Module,
        interval: int = 4,
        start_block: Optional[int] = None,   # 解冻起始 block 下标; None=最深一层 (blocks.N-1)
        num_layers: Optional[int] = None,    # 解冻层数; None=从起始层一直解冻到 blocks.0
    ):
        self.model = model
        self.interval = max(1, int(interval))          # 每 interval 个 epoch 解冻一层
        self.unfreeze_order: List[str] = self._build_order(start_block, num_layers)
        self.next_idx = 0
        self.unfreezed: List[str] = []
        self.unfreeze_log: List[Tuple[int, str]] = []

    def _build_order(self, start_block: Optional[int], num_layers: Optional[int]) -> List[str]:
        """按配置生成解冻顺序: 从 start_block 下标向下解冻 num_layers 层 (最深在前)。"""
        names = get_backbone_block_names(self.model)
        n = len(names)
        start = n - 1 if start_block is None else int(start_block)
        start = max(0, min(n - 1, start))              # 钳制到 [0, n-1]
        if num_layers is None:
            num = start + 1                            # 默认从 start 一直解冻到 blocks.0
        else:
            num = max(1, int(num_layers))
        num = min(num, start + 1)                      # 至多到 blocks.0
        return [f"blocks.{i}" for i in range(start, start - num, -1)]

    @property
    def active_layers(self) -> List[str]:
        """当前已解冻的层名列表 (累计)。"""
        return list(self.unfreezed)

    def apply_for_epoch(self, epoch: int, stage_name: str) -> List[str]:
        """每个 epoch 训练前调用。

        Stage2 中, 当 epoch 达到 (next_idx+1)*interval 时解冻下一个 block
        (从最后一个 block 开始往前)。Stage1 不做任何解冻。

        Returns:
            本次新解冻的层名列表 (空列表表示本 epoch 未解冻)
        """
        newly: List[str] = []
        if stage_name != "Stage2":
            return newly
        while (
            self.next_idx < len(self.unfreeze_order)
            and epoch >= (self.next_idx + 1) * self.interval
        ):
            block = self.unfreeze_order[self.next_idx]
            unfreeze_backbone_block(self.model, block)
            self.unfreezed.append(block)
            self.unfreeze_log.append((epoch, block))
            newly.append(block)
            self.next_idx += 1
        return newly

    def schedule_str(self) -> str:
        """生成解冻计划字符串, 如 'blocks.6@e2, blocks.5@e4', 用于最终报告。"""
        return ", ".join(f"{blk}@e{ep}" for ep, blk in self.unfreeze_log)
