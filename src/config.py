from pathlib import Path
from typing import Tuple, Union

import yaml
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _BaseValidatedConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')  # Disallow unexpected arguments.


class DataConfig(_BaseValidatedConfig):
    img_size: Tuple[int, int] = (224, 224)
    batch_size: int = 32
    data_split: Tuple[float, ...] = (0.7, 0.2, 0.1)
    num_workers: int = 0
    pin_memory: bool = True

    @model_validator(mode='after')
    def splits_add_up_to_one(self) -> 'DataConfig':
        epsilon = 1e-5
        total = sum(self.data_split)
        if abs(total - 1) > epsilon:
            raise ValueError(f'Splits should add up to 1, got {total}.')
        return self


class ExperimentConfig(_BaseValidatedConfig):
    project_name: str = 'ml_refactoring_lecture'
    experiment_name: str = 'image_classification'
    data_config: DataConfig = Field(default=DataConfig())

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> 'ExperimentConfig':
        cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        return cls(**cfg)

    def to_yaml(self, path: Union[str, Path]):
        with open(path, 'w') as out_file:
            yaml.safe_dump(self.model_dump(), out_file, default_flow_style=False, sort_keys=False)
