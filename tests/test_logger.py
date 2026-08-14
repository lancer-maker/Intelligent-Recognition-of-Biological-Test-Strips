import logging
from pathlib import Path

from src.training.trainer import StageTrainer
from src.utils.logger import setup_logger


def test_setup_logger_writes_to_file(tmp_path):
    log_file = tmp_path / "train.log"
    logger = setup_logger(str(log_file), level=logging.INFO)

    logger.info("hello logger")

    assert logger.name == "strip_reader"
    assert log_file.exists()
    assert "hello logger" in log_file.read_text(encoding="utf-8")


def test_stage_trainer_writes_loss_history_csv(tmp_path):
    trainer = StageTrainer.__new__(StageTrainer)
    trainer.loss_history = [
        {"fold_info": "1/24", "stage": "Stage1", "epoch": 1, "train_loss": 0.6, "val_loss": 0.7},
        {"fold_info": "1/24", "stage": "Stage2", "epoch": 1, "train_loss": 0.5, "val_loss": 0.4},
    ]

    trainer._save_loss_history("1/24", str(tmp_path))

    csv_path = Path(tmp_path) / "loss_history_1_24.csv"
    assert csv_path.exists()
    text = csv_path.read_text(encoding="utf-8")
    assert "train_loss" in text
    assert "0.6" in text
