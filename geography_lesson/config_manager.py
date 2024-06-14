from __future__ import annotations

from typing import Optional, Union
import os

from pydantic import BaseModel, validator, PydanticUserError
from torchvision import transforms
import numpy as np

class Parameters(BaseModel):
    batch_size: int
    augment: bool
    #transform: transforms.Compose | None
    name: str
    image_size: int
    patch_size: int
    num_classes: int
    dim: int
    depth: int
    heads: int
    mlp_dim: int
    distillation: Optional[bool] = False
    epochs: Optional[int] = 0
    max_data: Optional[int] = 1_000_000
    dropout: Optional[float] = 0
    emb_dropout: Optional[float] = 0
    teacher_name: Optional[str] = None
    teacher_image_size: Optional[int] = None
    teacher_patch_size: Optional[int] = None
    teacher_num_classes: Optional[int] = None
    teacher_dim: Optional[int] = None
    teacher_depth: Optional[int] = None
    teacher_heads: Optional[int] = None
    teacher_mlp_dim: Optional[int] = None
    teacher_dropout: Optional[float] = 0
    teacher_emb_dropout: Optional[float] = 0
    distill_temp: Optional[float] = 3
    distill_alpha: Optional[float] = 0.5
    learning_rate: Optional[float] = 0.001
    momentum: Optional[float] = 0.9
    nestrov: Optional[bool] = False
    weight_decay: Optional[float] = 0

class ConfigManager:
    def __init__(self) -> None:
        self.path = "geography_lesson/configs"
        self.configs = []
        for file in os.listdir(self.path):
            with open(os.path.join(self.path, file)) as f:
                cfg = Parameters.model_validate_json(f.read())
                self.configs.append(cfg)
        print(self.configs)

if __name__ == "__main__":
    params = Parameters(
        name="test2",
        max_data= 100_000,
        batch_size=32,
        augment=False,
        epochs=5,
        image_size=512,
        patch_size=32,
        num_classes=100,
        dim=512,
        depth=8,
        heads=6,
        mlp_dim=1024
    )
    # with open(f"geography_lesson/configs/{params.name}.json", "w") as f:
        # f.write(params.schema_json(indent=2))
    cfgm = ConfigManager()