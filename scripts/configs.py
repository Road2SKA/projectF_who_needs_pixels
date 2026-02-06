from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
import yaml


class PathsConfig(BaseModel):
    data: str = "data/meerkat_patch.npz"
    model: str = "data/siren_meerkat.pt"
    fits: str = "MeerKAT_Galactic_Centre_1284MHz-StokesI.fits"
    checkpoints_dir: str = "data/checkpoints"
    images_dir: str = "data"
    tune_dir: str = "tune_runs"


class ModelConfig(BaseModel):
    in_features: int = 3
    out_features: int = 1
    hidden_features: int = 256
    hidden_layers: int = 3
    outermost_linear: bool = True
    first_omega: float = 0.1
    hidden_omega: float = 0.1
    lambda_wavelet: float = 0.001


class PrepareConfig(BaseModel):
    patch_height: int | None = None
    patch_width: int | None = None
    x0: int | None = None
    y0: int | None = None
    out: str | None = None
    dtype: str = "float32"
    debug_plots: bool = False
    meta_path: str | None = None


class CommonConfig(BaseModel):
    batch_size: int = 512
    lr: float = 1e-4
    amp: str = "fp16"
    data_dtype: str = "float32"
    tf32: bool = False
    compile: bool = False


class TrainConfig(BaseModel):
    device: Literal["cpu", "cuda", "mps", "auto"] = "auto"
    steps: int = 500
    out: str | None = None
    checkpoint_every: int = 0
    save_image_every: int = 0
    early_stop: bool = False
    early_stop_every: int = 0
    early_stop_patience: int = 3
    early_stop_tolerance: float = 1e-6
    wavelet: str = "db1"
    level: int = 2
    mode: str = "constant"
    omega_double_on: int = 50


class InferenceConfig(BaseModel):
    batch_size: int | None = None
    title_prefix: str = ""
    save_plot: str | None = None
    no_plot: bool = False


class TuneConfig(BaseModel):
    seed: int | None = None
    n_trials: int = 20
    steps: int = 500
    early_stop: bool = False
    early_stop_every: int = 0
    early_stop_patience: int = 3
    early_stop_tolerance: float = 1e-6
    search: dict[str, Any] = Field(default_factory=dict)
    fixed: dict[str, Any] = Field(default_factory=dict)


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    paths: PathsConfig = Field(default_factory=PathsConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    prepare: PrepareConfig = Field(default_factory=PrepareConfig)
    common: CommonConfig = Field(default_factory=CommonConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    tune: TuneConfig = Field(default_factory=TuneConfig)

    def to_json(self, **kwargs: Any) -> str:
        return self.model_dump_json(**kwargs)


def _apply_common(raw: dict[str, Any]) -> dict[str, Any]:
    return raw


def load_config(path: Path) -> AppConfig:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError("Config must be a mapping")
    raw = _apply_common(raw)
    return AppConfig.model_validate(raw)
