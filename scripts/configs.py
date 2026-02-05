from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
import yaml


class PathsConfig(BaseModel):
    data: str = "data/meerkat_patch.npz"
    model: str = "data/siren_meerkat.pt"
    fits: str = "MeerKAT_Galactic_Centre_1284MHz-StokesI.fits"


class ModelConfig(BaseModel):
    in_features: int = 3
    out_features: int = 1
    hidden_features: int = 256
    hidden_layers: int = 3
    outermost_linear: bool = True
    first_omega: float = 0.1
    hidden_omega: float = 0.1


class PrepareConfig(BaseModel):
    patch_height: int | None = None
    patch_width: int | None = None
    x0: int | None = None
    y0: int | None = None
    out: str | None = None
    dtype: str = "float32"
    debug_plots: bool = False


class TrainConfig(BaseModel):
    device: Literal["cpu", "cuda", "mps", "auto"] = "auto"
    steps: int = 500
    batch_size: int = 512
    lr: float = 1e-4
    out: str | None = None
    amp: str = "fp16"
    data_dtype: str = "float32"
    tf32: bool = False
    compile: bool = False
    lambda_wavelet: float = 0.001
    wavelet: str = "db1"
    level: int = 2
    mode: str = "constant"


class InferenceConfig(BaseModel):
    batch_size: int = 4096
    title_prefix: str = ""
    save_plot: str | None = None
    no_plot: bool = False


class TuneConfig(BaseModel):
    out_dir: str = "tune_runs"
    seed: int | None = None
    n_trials: int = 20
    steps: int = 500
    batch_size: int = 512
    lr: float = 1e-4
    amp: str = "fp16"
    data_dtype: str = "float32"
    compile: bool = False
    tf32: bool = False
    search: dict[str, Any] = Field(default_factory=dict)
    fixed: dict[str, Any] = Field(default_factory=dict)


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    paths: PathsConfig = Field(default_factory=PathsConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    prepare: PrepareConfig = Field(default_factory=PrepareConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    tune: TuneConfig = Field(default_factory=TuneConfig)

    def to_json(self, **kwargs: Any) -> str:
        return self.model_dump_json(**kwargs)


def _apply_common(raw: dict[str, Any]) -> dict[str, Any]:
    common = raw.pop("common", None)
    if not isinstance(common, dict):
        return raw

    paths = raw.get("paths", {})
    model = raw.get("model", {})

    for key, value in common.items():
        if key in PathsConfig.model_fields:
            paths.setdefault(key, value)
        elif key in ModelConfig.model_fields:
            model.setdefault(key, value)

    if paths:
        raw["paths"] = paths
    if model:
        raw["model"] = model
    return raw


def load_config(path: Path) -> AppConfig:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError("Config must be a mapping")
    raw = _apply_common(raw)
    return AppConfig.model_validate(raw)
