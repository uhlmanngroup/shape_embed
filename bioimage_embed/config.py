"""
Configuration for bie works by using pydantic dataclasses to define the 
configuration schema. This is then fed into hydra to generate the 
configuration files. The configuration schema is defined in the `Config` 
class. The `Config` class is a dataclass that contains other dataclasses 
as fields. Each of these dataclasses define a specific part of the 
configuration schema. This master Config is then wrapped by the BioImageEmbed 
Class to run model training and inferece. Config is given sane defaults for 
autoencoding Pydantic (in future) will be further used to validate the 
configuration schema, so that the configuration files generated by hydra 
are valid.

"""

from bioimage_embed.augmentations import (
    DEFAULT_ALBUMENTATION,
)
import os
from pydantic.dataclasses import dataclass
from typing import List, Optional, Dict, Any
from types import SimpleNamespace
from pydantic import BaseModel, Field, field_validator, root_validator
from omegaconf import SI, II
from . import utils


@dataclass
class Recipe:
    _target_: str = "types.SimpleNamespace"
    model: str = "resnet18_vae"
    data: str = "data"
    opt: str = "adamw"
    max_epochs: int = 125
    weight_decay: float = 0.001
    momentum: float = 0.9
    sched: str = "cosine"
    epochs: int = 50
    lr: float = 1e-4
    min_lr: float = 1e-6
    t_initial: int = 10
    t_mul: int = 2
    lr_min: Optional[float] = None
    decay_rate: float = 0.1
    warmup_lr: float = 1e-6
    warmup_lr_init: float = 1e-6
    warmup_epochs: int = 5
    cycle_limit: Optional[int] = None
    t_in_epochs: bool = False
    noisy: bool = False
    noise_std: float = 0.1
    noise_pct: float = 0.67
    noise_seed: Optional[int] = None
    cooldown_epochs: int = 5
    warmup_t: int = 0
    seed: int = 42


# Use the ALbumentations .to_dict() method to get the dictionary
# that pydantic can use
@dataclass
class ATransform:
    _target_: str = "albumentations.from_dict"
    _convert_: str = "object"
    # _convert_: str = "all"
    transform_dict: Dict = Field(
        default_factory=lambda: DEFAULT_ALBUMENTATION.to_dict()
    )


# VisionWrapper is a helper class for applying albumentations pipelines for image augmentations in autoencoding


@dataclass
class Transform:
    _target_: str = "bioimage_embed.augmentations.VisionWrapper"
    _convert_: str = "object"
    # transform: ATransform = field(default_factory=ATransform)
    transform_dict: Dict = Field(
        default_factory=lambda: DEFAULT_ALBUMENTATION.to_dict()
    )


@dataclass
class Dataset:
    # _target_: str = "torch.utils.data.Dataset"
    transform: Transform = Field(default_factory=Transform)


@dataclass
class ImageFolderDataset(Dataset):
    _target_: str = "torchvision.datasets.ImageFolder"
    # transform: Transform = Field(default_factory=Transform)
    root: str = II("recipe.data")


@dataclass
class NdDataset(ImageFolderDataset):
    transform: Transform = Field(default_factory=Transform)


@dataclass
class TiffDataset(NdDataset):
    _target_: str = "bioimage_embed.datasets.TiffDataset"


class NgffDataset(NdDataset):
    _target_: str = "bioimage_embed.datasets.NgffDataset"


@dataclass
class DataLoader:
    _target_: str = "bioimage_embed.lightning.dataloader.DataModule"
    dataset: ImageFolderDataset = Field(default_factory=ImageFolderDataset)
    num_workers: int = 1

@dataclass
class Model:
    _target_: str = "bioimage_embed.models.create_model"
    model: str = II("recipe.model")
    input_dim: List[int] = Field(default_factory=lambda: [3, 224, 224])
    latent_dim: int = 64
    pretrained: bool = True


@dataclass
class Callback:
    pass


@dataclass
class EarlyStopping(Callback):
    _target_: str = "pytorch_lightning.callbacks.EarlyStopping"
    monitor: str = "loss/val"
    mode: str = "min"
    patience: int = 3


@dataclass
class ModelCheckpoint(Callback):
    _target_: str = "pytorch_lightning.callbacks.ModelCheckpoint"
    save_last = True
    save_top_k = 1
    monitor = "loss/val"
    mode = "min"
    # dirpath: str = Field(default_factory=lambda: utils.hashing_fn(Recipe()))
    dirpath: str = f"{II('paths.model')}/{II('uuid')}"


@dataclass
class LightningModel:
    _target_: str = "bioimage_embed.lightning.torch.LitAutoEncoderTorch"
    # This should be pythae base autoencoder?
    model: Model = Field(default_factory=Model)
    args: Recipe = Field(default_factory=lambda: II("recipe"))


@dataclass
class Callbacks:
    # _target_: str = "collections.OrderedDict"
    model_checkpoint: ModelCheckpoint = Field(default_factory=ModelCheckpoint)
    early_stopping: EarlyStopping = Field(default_factory=EarlyStopping)


@dataclass
class Trainer:
    _target_: str = "pytorch_lightning.Trainer"
    # logger: Optional[any]
    gradient_clip_val: float = 0.5
    enable_checkpointing: bool = True
    devices: str = "auto"
    accelerator: str = "auto"
    accumulate_grad_batches: int = 16
    min_epochs: int = 1
    max_epochs: int = II("recipe.max_epochs")
    log_every_n_steps: int = 1
    # This is not a clean implementation but I am not sure how to do it better
    callbacks: List[Any] = Field(
        default_factory=lambda: list(vars(Callbacks()).values()), frozen=True
    )


# TODO add argument caching for checkpointing


@dataclass
class Paths:
    model: str = "models"
    logs: str = "logs"
    tensorboard: str = "tensorboard"
    wandb: str = "wandb"

    @root_validator(
        pre=False, skip_on_failure=True
    )  # Ensures this runs after all other validations
    @classmethod
    def create_dirs(cls, values):
        # The `values` dict contains all the validated field values
        for path in values.values():
            os.makedirs(path, exist_ok=True)
        return values


@dataclass
class Config:
    paths: Paths = Field(default_factory=Paths)
    recipe: Recipe = Field(default_factory=Recipe)
    dataloader: DataLoader = Field(default_factory=DataLoader)
    trainer: Trainer = Field(default_factory=Trainer)
    lit_model: LightningModel = Field(default_factory=LightningModel)
    # callbacks: Callbacks = field(default_factory=Callbacks)
    uuid: str = Field(default_factory=lambda: utils.hashing_fn(Recipe()))


__schemas__ = {
    "recipe": Recipe,
    "transform": Transform,
    "dataset": ImageFolderDataset,
    "dataloader": DataLoader,
    "trainer": Trainer,
    "model": Model,
    "lit_model": LightningModel,
}
