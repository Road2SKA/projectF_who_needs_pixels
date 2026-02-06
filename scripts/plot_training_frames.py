#!/usr/bin/env python3
import argparse
from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
import numpy as np, json
from pathlib import Path


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



def save_image(image: np.ndarray, out_path: Path, cmap: str, vmin, vmax, dpi: int):
    fig, ax = plt.subplots(figsize=(5, 5), dpi=dpi)
    ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot train_pred_step_*.npy images and save as PNGs"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing train_pred_step_*.npy files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/frames"),
        help="Directory to save plotted PNGs",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="train_pred_step_*.npy",
        help="Glob pattern for input files",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="gray",
        help="Matplotlib colormap",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="Lower value for colormap scaling",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Upper value for colormap scaling",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        nargs=2,
        default=None,
        metavar=("LOW", "HIGH"),
        help="Percentile range for vmin/vmax (e.g. 1 99)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output DPI",
    )
    parser.add_argument(
        "--meta-path",
        type=str,
        default="norm_info.json",
        help="Path to normalization metadata JSON file",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob(args.pattern), key=_extract_step)
    if not files:
        raise FileNotFoundError(
            f"No files found in {input_dir} matching {args.pattern}"
        )

    for npy_path in files:
        image = np.load(npy_path)

        image = undo_normalize_log_meerkat(image, args.meta_path)
        vmin, vmax = _compute_vmin_vmax(image, args.vmin, args.vmax, args.percentile)
        out_name = f"{npy_path.stem}.png"
        out_path = output_dir / out_name
        save_image(image, out_path, args.cmap, vmin, vmax, args.dpi)


if __name__ == "__main__":
    main()
