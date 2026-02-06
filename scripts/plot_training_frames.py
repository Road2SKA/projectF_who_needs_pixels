#!/usr/bin/env python3
import argparse
from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
import numpy as np, json
from pathlib import Path
import copy
from matplotlib.colors import Normalize
import json

def _extract_step(path: Path) -> int:
    match = re.search(r"train_pred_step_(\d+)", path.stem)
    return int(match.group(1)) if match else -1


def _compute_vmin_vmax(image: np.ndarray, vmin, vmax, percentile):
    if vmin is not None or vmax is not None:
        return vmin, vmax
    if percentile is not None:
        lo, hi = percentile
        return (
            float(np.percentile(image, lo)),
            float(np.percentile(image, hi)),
        )
    return None, None


def undo_normalize_log_meerkat(norm_data, meta_path):

    meta_path = Path(meta_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)

    min1 = meta["min1"]
    eps  = meta["eps"]
    min2 = meta["min2"]
    max2 = meta["max2"]

    # Undo [-1,1] → [0,1]
    norm01 = (norm_data + 1.0) / 2.0

    data3 = norm01 * max2 + min2
    data2 = np.exp(data3)
    data1 = data2 - eps
    data  = data1 + min1

    return data

class ThresholdSqrtNorm(Normalize):
    """
    Map values <= vmin to "under" (so cmap.set_under works),
    and apply sqrt stretch for values in [vmin, vmax].
    """

    def __call__(self, value, clip=None):
        v = np.array(value, copy=False)

        result = np.ma.array(v, copy=True)
        mask = ~np.isfinite(result)
        result = np.ma.masked_where(mask, result)

        vmin, vmax = self.vmin, self.vmax
        if vmin is None or vmax is None:
            raise ValueError("vmin and vmax must be set")

        scaled = (result - vmin) / (vmax - vmin)
        scaled = np.ma.where(result >= vmin, np.sqrt(np.clip(scaled, 0, 1)), scaled)

        return scaled

def save_image(image: np.ndarray, out_path: Path, cmap: str, vmin, vmax, dpi: int):
    fig, ax = plt.subplots(figsize=(5, 5), dpi=dpi)
    ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_image_dual(
    image: np.ndarray,
    out_path: Path,
    *,
    gray_cmap: str,
    heat_cmap: str,
    low_vmin: float,
    low_vmax: float,
    threshold: float,
    high_vmax: float,
    dpi: int,
):
    """
    Dual-layer rendering:
      1) faint grayscale (linear): [low_vmin, low_vmax]
      2) bright heat overlay (sqrt): [threshold, high_vmax], transparent under threshold
    """
    fig, ax = plt.subplots(figsize=(5, 5), dpi=dpi)

    # Faint layer
    ax.imshow(
        image,
        origin="lower",
        cmap=gray_cmap,
        vmin=low_vmin,
        vmax=low_vmax,
    )

    # Heat layer with transparency under threshold
    heat = copy.copy(plt.get_cmap(heat_cmap))
    heat.set_under((0, 0, 0, 0))  # fully transparent for values < threshold

    norm_hi = ThresholdSqrtNorm(vmin=threshold, vmax=high_vmax)
    ax.imshow(
        image,
        origin="lower",
        cmap=heat,
        norm=norm_hi,
    )

    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot train_pred_step_*.npy images and save as PNGs"
    )
    parser.add_argument("--input-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/frames"))
    parser.add_argument("--pattern", type=str, default="train_pred_step_*.npy")

    parser.add_argument("--dpi", type=int, default=150)

    parser.add_argument(
        "--meta-path",
        type=str,
        default="norm_info.json",
        help="Path to normalization metadata JSON file",
    )

    # --- Single-layer options ---
    parser.add_argument("--cmap", type=str, default="gray")
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument(
        "--percentile",
        type=float,
        nargs=2,
        default=None,
        metavar=("LOW", "HIGH"),
        help="Percentile range for vmin/vmax (e.g. 1 99)",
    )

    # --- Dual-layer options ---
    parser.add_argument(
        "--dual",
        action="store_true",
        help="Enable dual colour scaling (grayscale faint + heat overlay with sqrt stretch).",
    )
    parser.add_argument("--gray-cmap", type=str, default="gray_r")
    parser.add_argument("--heat-cmap", type=str, default="afmhot")

    # Defaults from your example (Heywood 2022 values):
    parser.add_argument("--threshold", type=float, default=0.4e-3)
    parser.add_argument("--low-vmin", type=float, default=-0.4e-3)
    parser.add_argument("--low-vmax", type=float, default=0.4e-3)
    parser.add_argument("--high-vmax", type=float, default=350e-3)

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob(args.pattern), key=_extract_step)
    if not files:
        raise FileNotFoundError(f"No files found in {input_dir} matching {args.pattern}")

    for npy_path in files:
        image = np.load(npy_path)

        # Undo normalization back to original units
        image = undo_normalize_log_meerkat(image, args.meta_path)

        out_path = output_dir / f"{npy_path.stem}.png"

        if args.dual:
            save_image_dual(
                image,
                out_path,
                gray_cmap=args.gray_cmap,
                heat_cmap=args.heat_cmap,
                low_vmin=args.low_vmin,
                low_vmax=args.low_vmax,
                threshold=args.threshold,
                high_vmax=args.high_vmax,
                dpi=args.dpi,
            )
        else:
            vmin, vmax = _compute_vmin_vmax(image, args.vmin, args.vmax, args.percentile)
            save_image(image, out_path, args.cmap, vmin, vmax, args.dpi)


if __name__ == "__main__":
    main()