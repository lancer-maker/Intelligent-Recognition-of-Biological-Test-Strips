from pathlib import Path
import yaml


def get_main_config(config_path: str = "configs/main_config.yaml") -> dict:
    """加载项目主配置文件，默认读取仓库根目录下的 config.yaml。"""
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = Path(__file__).resolve().parents[1] / config_file

    if not config_file.exists():
        return {}

    with config_file.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def get_aug_config(config_path: str = "configs/augmentation_presets.yaml") -> dict:
    """加载项目主配置文件，默认读取仓库根目录下的 config.yaml。"""
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = Path(__file__).resolve().parents[1] / config_file

    if not config_file.exists():
        return {}

    with config_file.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}