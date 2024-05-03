from dataclasses import dataclass
from typing import Literal

@dataclass
class Model:
    name: str
    image_size: int
    patch_size: int
    in_channels: int
    layers: int
    heads: int
    embedding_dim: int
    mlp_ratio: int
    num_output_classes: int
    norm_type: Literal["prenorm", "postnorm"]
    dropout: float
    qkv_bias: bool

@dataclass
class Data:
    train_data_dir:str
    valid_data_dir:str
    test_data_dir:str
    train_labels_pickle:str
    valid_labels_pickle:str
    test_labels_pickle:str

@dataclass
class Scheduler:
    name: str

@dataclass
class Scheduler_ReduceOnPlateau(Scheduler):
    patience: int
    factor: float

@dataclass
class Scheduler_OneCycleLR(Scheduler):
    expand_lr: int

@dataclass
class Training:
    project_name: str
    train_name: str
    seed: int
    epochs: int
    batch: int
    lr: float
    wandb_path: str
    model_path: str
    save_best_of: int
    checkpoint_monitor: str
    early_stopping_patience: int

@dataclass
class Params:
    model: Model
    data: Data
    training: Training
    scheduler: Scheduler