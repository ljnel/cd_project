import numpy as np
from pathlib import Path
from datetime import datetime
import inspect
import matplotlib.pyplot as plt

def get_project_root() -> Path:
    path = Path(__file__).resolve()
    for parent in [path] + list(path.parents):
        if parent.name == "cd_project" and (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("Could not find cd_project root with pyproject.toml")

def get_log_dir() -> str:
    root = get_project_root()
    return str(root / "outputs" / "logs")

def get_ckpt_dir() -> str:
    root = get_project_root()
    return str(root / "outputs" / "checkpoints")

def save_plot(
    name: str,
    ax=None,
    *,
    subfolder: str = "outputs/plots",
    ext: str = "png",
    dpi: int = 300,
    bbox_inches: str | None = "tight"
) -> Path:
    caller_stem = Path(inspect.stack()[1].filename).stem
    base_stem   = f"{caller_stem}_{name}"

    project_root = get_project_root()

    output_dir = project_root / subfolder
    output_dir.mkdir(parents=True, exist_ok=True)

    outfile = output_dir / f"{base_stem}.{ext}"

    fig = ax.figure if ax is not None else plt.gcf()
    fig.savefig(outfile, dpi=dpi, bbox_inches=bbox_inches)
    print(f"Plot saved to: {outfile}")
    return outfile
