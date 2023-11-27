import os

import lightning

from src.config import ExperimentConfig
from src.constants import PROJECT_ROOT
from src.datamodule import ClassificationDataModule
from src.logger import LOGGER


def train(cfg: ExperimentConfig):
    lightning.seed_everything(0)
    datamodule = ClassificationDataModule(cfg=cfg.data_config)

    datamodule.prepare_data()
    datamodule.setup(stage='fit')
    img, labels = next(iter(datamodule.train_dataloader()))
    LOGGER.debug('Got batch of images with shape %s', img.shape)


if __name__ == '__main__':
    cfg_path = os.getenv('TRAIN_CFG_PATH', PROJECT_ROOT / 'configs' / 'train.yaml')
    train(cfg=ExperimentConfig.from_yaml(cfg_path))
