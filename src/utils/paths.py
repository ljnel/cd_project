import inspect
from pathlib import Path


def get_root() -> Path:
    path = Path(__file__).resolve()
    for parent in [path] + list(path.parents):
        if parent.name == "cd_project" and (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("Could not find cd_project root with pyproject.toml")


def get_output_dir(*parts: str) -> Path:
    """Return outputs/<calling-script>/<parts...>, creating it if needed."""
    name = Path(inspect.stack()[1].filename).stem
    out = get_root().joinpath("outputs", name, *parts)
    out.mkdir(parents=True, exist_ok=True)
    return out
