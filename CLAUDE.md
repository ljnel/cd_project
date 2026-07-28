# CD Project

Anomaly detection for robot trajectories using the Christoffel-Darboux polynomial and its kernelized variant.

## Setup

- Package manager: pixi (`pixi install`, `pixi shell`)
- Source root: `cd/` (flat layout; editable install via pyproject.toml)
- Tests: `pixi run pytest tests/`
- Lint: `pixi run ruff check cd`

## Script convention

- Scripts define a `main()` whose typed signature is the CLI (parsed via `tyro.cli(main)`, not `argparse`), with the experiment and args documented in `main`'s docstring instead of a module-level one.
- Whenever you are asked to create a script that produces a plot, use Mac's "open" tool to show it to me.

## Output convention

Every script in `scripts/` writes its artifacts under `outputs/<script-stem>/`.

- Get the directory with `get_output_dir(*parts)` from `cd.utils.paths` — returns
  `outputs/<calling-script-filename-stem>/<parts...>` and creates it. Build file
  paths by appending: `get_output_dir() / "roc.pdf"`, `get_output_dir("curves") / f"{env}.pdf"`.
- Save figures with `save_plot(path, *, fig=None, dpi=300, bbox_inches="tight")`
  from `cd.utils.plotting` (defaults to PDF if `path` has no extension).
- Never hardcode paths, write into the source tree, or use CWD-relative paths.
- Folder = script stem, so a consumer reading another script's output references
  the producer explicitly, e.g. `get_root() / "outputs" / "tune" / f"{env}.json"`.
- `outputs/` is gitignored and disposable (regenerable). `scripts/sync_figures.sh`
  pushes `*.pdf`/`*.tex` from `outputs/` to Overleaf.
- Exception: caches shared across scripts use one shared dir (e.g. the hopper
  scripts share `get_root() / "outputs" / "hopper_cache"`).
