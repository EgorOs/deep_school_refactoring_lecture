from pathlib import Path

from src.config import ExperimentConfig
from src.constants import PROJECT_ROOT
from src.train import train


def test_training_pipeline():
    """Ensure train loop is running end-to-end."""
    # Given
    cfg_path = PROJECT_ROOT / 'configs' / 'train.yaml'
    cfg = ExperimentConfig.from_yaml(cfg_path)

    cfg.data_config.data_path = Path('tests') / 'assets' / 'sample_dataset' / 'Classification_data'  # noqa: WPS221
    cfg.trainer_config.fast_dev_run = True
    cfg.data_config.batch_size = 2
    cfg.track_in_clearml = False  # Prevent test from logging to ClearML

    # When
    train(cfg=cfg)
