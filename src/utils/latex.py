
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from config.datasets import DATASETS
from config.detectors import DEFAULT_METHODS, METHOD_GROUP_BREAKS, get_method_display_name
from config.envs import ENV_INFO
from config.tasks import TASK_CONFIGS
from data.datasets import load_dataset
from utils.paths import get_root

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_env_display_name(env_key: str) -> str:
    """Get display name for an environment from config."""
    info = ENV_INFO.get(env_key)
    return info.display_name if info else env_key


def compute_avg_ranks(
    all_results: dict[str, dict],
    metric_key: str,
    methods: list[str],
    envs: list[str],
) -> dict[str, float]:
    """Compute average rank of each method across environments (lower is better).

    Rank 1 = best. Ties get the average of tied ranks.
    """
    # (n_envs, n_methods) matrix of metric values
    # Handle both scalar and (mean, std) tuple formats
    def _extract_mean(val):
        return val[0] if isinstance(val, tuple) else val

    scores = np.array([
        [_extract_mean(all_results[env]['results'][method][metric_key]) for method in methods]
        for env in envs
    ])
    # Rank per env (higher metric = rank 1), then average across envs
    ranks = np.array([rankdata(-row, method='average') for row in scores])
    avg_ranks = ranks.mean(axis=0)
    return {method: avg_ranks[i] for i, method in enumerate(methods)}


# ---------------------------------------------------------------------------
# Per-experiment formatting functions
# ---------------------------------------------------------------------------

def format_metric_latex_table(
    all_results: dict[str, dict],
    metric_key: str,
    caption: str,
    label: str,
) -> str:
    """
    Format a single scalar metric as a LaTeX table with environments as columns.
    The best method per environment is bolded.
    """
    envs = list(all_results.keys())
    methods = list(next(iter(all_results.values()))['results'].keys())

    def _extract_mean(val):
        return val[0] if isinstance(val, tuple) else val

    # Find best method for each environment (by mean)
    best_method_per_env = {}
    for env in envs:
        best_val = -1
        best_method = None
        for method in methods:
            val = _extract_mean(all_results[env]['results'][method][metric_key])
            if val > best_val:
                best_val = val
                best_method = method
        best_method_per_env[env] = best_method

    # Build table header
    col_spec = "@{}l" + "c" * len(envs) + "c@{}"
    env_display_names = [get_env_display_name(env) for env in envs]
    header_row = " & ".join(f"\\textbf{{{n}}}" for n in env_display_names)

    latex = (
        f"\\begin{{table*}}[htbp]\n"
        f"\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{tab:{label}}}\n"
        f"\\begin{{tabular}}{{{col_spec}}}\n"
        f"\\toprule\n"
        f"\\textbf{{Method}} & {header_row} & \\textbf{{Avg. Rank}} \\\\\n"
        f"\\midrule\n"
    )

    # Compute average ranks across environments
    avg_ranks = compute_avg_ranks(all_results, metric_key, methods, envs)
    best_rank_method = min(avg_ranks, key=avg_ranks.get)

    # Add rows for each method
    for i, method in enumerate(methods):
        if i in METHOD_GROUP_BREAKS:
            latex += "\\midrule\n"
        cells = [method]
        for env in envs:
            raw = all_results[env]['results'][method][metric_key]
            if isinstance(raw, tuple):
                mean, std = raw
                cell = f"{mean:.3f} $\\pm$ {std:.3f}"
            else:
                cell = f"{raw:.3f}"
            if method == best_method_per_env[env]:
                cell = f"\\textbf{{{cell}}}"
            cells.append(cell)
        rank_val = f"{avg_ranks[method]:.1f}"
        if method == best_rank_method:
            rank_val = f"\\textbf{{{rank_val}}}"
        cells.append(rank_val)
        latex += " & ".join(cells) + " \\\\\n"

    latex += (
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table*}\n"
    )
    return latex


def format_compute_cost_table(
    results: dict[str, dict[str, float]],
    env_name: str,
) -> str:
    """Format computational cost results as a LaTeX table."""
    display_name = get_env_display_name(env_name)

    latex = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        f"\\caption{{Computational cost on {display_name.lower()}.}}\n"
        f"\\label{{tab:{env_name}_compute}}\n"
        "\\begin{tabular}{@{}lccc@{}}\n"
        "\\toprule\n"
        "\\textbf{Method} & \\textbf{Train (s)}"
        " & \\textbf{Predict Total (ms)}"
        " & \\textbf{Predict/Sample (ms)} \\\\\n"
        "\\midrule\n"
    )

    for method, m in results.items():
        train = f"{m['train_time_mean']:.2f} $\\pm$ {m['train_time_std']:.2f}"
        pred = f"{m['predict_time_mean']:.1f} $\\pm$ {m['predict_time_std']:.1f}"
        per_sample = (f"{m['predict_per_sample_mean']:.3f} $\\pm$ "
                      f"{m['predict_per_sample_std']:.3f}")
        latex += f"{method} & {train} & {pred} & {per_sample} \\\\\n"

    latex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    return latex


# ---------------------------------------------------------------------------
# Environment summary table
# ---------------------------------------------------------------------------

def _format_range(rng: tuple[float, float]) -> str:
    """Format a domain-randomization range for LaTeX display."""
    lo, hi = rng
    if lo == hi:
        return "---"
    return f"[{lo}, {hi}]"


def format_env_summary_table(
    env_keys: list[str],
    rows: list[dict],
) -> str:
    """Format an environment summary as a LaTeX table.

    Each row dict contains: display_name, W, H, obs_dim,
    mass_range, friction_range, damping_range, failure_prop (float or None),
    term_cond, term_note, term_custom.
    """
    n_envs = len(env_keys)
    assert len(rows) == n_envs

    latex = (
        "\\begin{table*}[htbp]\n"
        "\\centering\n"
        "\\caption{Setup summary for each environment, showing window length,\n"
        "horizon length, observation space dimension, ranges for domain\n"
        "randomization, proportion of failed episodes, and termination condition.}\n"
        "\\label{tab:env_summary}\n"
        "\\begin{tabular}{@{}lcccccccp{3.8cm}@{}}\n"
        "\\toprule\n"
        "\\textbf{Environment} & \\textbf{$W$} & \\textbf{$H$}"
        " & \\textbf{Obs dim} & \\textbf{Mass} & \\textbf{Friction}"
        " & \\textbf{Damping} & \\textbf{Fail.\\ prop.}"
        " & \\textbf{Termination} \\\\\n"
        "\\midrule\n"
    )

    for row in rows:
        fp = f"{row['failure_prop']:.3f}" if row['failure_prop'] is not None else "---"
        star = "*" if row['term_custom'] else ""
        term_cell = f"{row['term_cond']}{star} {{\\scriptsize ({row['term_note']})}}"
        latex += (
            f"{row['display_name']} & {row['W']} & {row['H']} & {row['obs_dim']} "
            f"& {_format_range(row['mass_range'])} "
            f"& {_format_range(row['friction_range'])} "
            f"& {_format_range(row['damping_range'])} "
            f"& {fp} & {term_cell} \\\\\n"
        )

    latex += "\\bottomrule\n\\end{tabular}\n"
    if any(r['term_custom'] for r in rows):
        latex += ("\\smallskip\\\\\n"
                  "{\\scriptsize *Custom termination condition"
                  " (all others are environment defaults).}\n")
    latex += "\\end{table*}\n"
    return latex



def generate_env_summary_table(output_dir: Path):
    """Generate environment summary LaTeX tables from configs and dataset files."""
    env_keys = list(TASK_CONFIGS.keys())
    rows = []

    for env in env_keys:
        task_cfg = TASK_CONFIGS[env]
        env_info = ENV_INFO[env]
        ds_key = f"{env}/test"
        ds_cfg = DATASETS.get(ds_key)

        # Domain randomization ranges
        mass_range = ds_cfg.mass_range if ds_cfg else (1.0, 1.0)
        friction_range = ds_cfg.friction_range if ds_cfg else (1.0, 1.0)
        damping_range = ds_cfg.damping_range if ds_cfg else (1.0, 1.0)

        # Failure proportion from dataset on disk
        failure_prop = None
        if ds_cfg:
            try:
                data = load_dataset(ds_cfg)
                fail = data['fail']
                failure_prop = float(np.sum(fail > -1)) / len(fail)
            except FileNotFoundError:
                pass

        rows.append({
            'display_name': env_info.display_name,
            'W': task_cfg.win,
            'H': task_cfg.hor,
            'obs_dim': (env_info.obs_slice.stop - (env_info.obs_slice.start or 0)
                       if env_info.obs_slice is not None else env_info.obs_dim),
            'mass_range': mass_range,
            'friction_range': friction_range,
            'damping_range': damping_range,
            'failure_prop': failure_prop,
            'term_cond': env_info.term_cond,
            'term_note': env_info.term_note,
            'term_custom': env_info.term_custom,
        })

    output_dir.mkdir(parents=True, exist_ok=True)

    tex = format_env_summary_table(env_keys, rows)
    tex_file = output_dir / "env_summary.tex"
    tex_file.write_text(tex)
    print(f"  {tex_file}")


# ---------------------------------------------------------------------------
# Evaluation (score quality + deployment quality) formatting
# ---------------------------------------------------------------------------

def _load_evaluation_results(results_dir: Path) -> dict[str, dict]:
    """Load evaluation .npz files into a structured dict.

    Expects per-method files at ``results_dir/{env}/{method}.npz``.

    Returns {env: {method_key: {fpr, det_rate, med_ttd}}}
    """
    all_results = {}
    for env_dir in sorted(results_dir.iterdir()):
        if not env_dir.is_dir():
            continue
        env_name = env_dir.name
        env_results = {}
        for npz_file in sorted(env_dir.glob("*.npz")):
            data = np.load(npz_file, allow_pickle=True)
            method = npz_file.stem
            env_results[method] = {
                'fpr': float(data.get('fpr', np.nan)),
                'det_rate': float(data.get('det_rate', np.nan)),
                'med_ttd': float(data.get('med_ttd', np.nan)),
            }
        if env_results:
            all_results[env_name] = env_results
    return all_results


def format_evaluation_table(
    all_results: dict[str, dict],
) -> str:
    """Format deployment metrics across all environments as a single LaTeX table.

    Methods as rows, environments as column groups with FPR/Det/TTD sub-columns.
    """
    envs = list(all_results.keys())
    methods = list(next(iter(all_results.values())).keys())
    display_methods = [get_method_display_name(m) for m in methods]
    env_display = [get_env_display_name(e) for e in envs]

    metrics = [
        ('det_rate', 'EDR (\\%)', '{:.1f}', False),
        ('med_ttd', 'TTD (\\%)', '{:.1f}', False),
    ]
    n_met = len(metrics)

    # Column spec: method name + 3 sub-columns per env
    col_spec = "@{}l" + "ccc" * len(envs) + "@{}"

    # Header row 1: env names spanning 3 columns each
    env_headers = " & ".join(
        f"\\multicolumn{{{n_met}}}{{c}}{{\\textbf{{{name}}}}}"
        for name in env_display
    )

    # Header row 2: metric sub-headers repeated per env
    sub_headers = " & ".join(
        " & ".join(f"\\textbf{{{label}}}" for _, label, _, _ in metrics)
        for _ in envs
    )

    # cmidrules under each env group
    cmidrules = ""
    for i in range(len(envs)):
        start = 2 + i * n_met
        end = start + n_met - 1
        cmidrules += f"\\cmidrule(lr){{{start}-{end}}} "

    # Best per env per metric
    best_per = {}
    for env in envs:
        for mkey, _, _, lower_better in metrics:
            best_val = np.inf if lower_better else -np.inf
            best_m = None
            for m in methods:
                v = all_results[env][m][mkey]
                if np.isnan(v):
                    continue
                if (lower_better and v < best_val) or (not lower_better and v > best_val):
                    best_val = v
                    best_m = m
            best_per[(env, mkey)] = best_m

    latex = (
        "\\begin{table*}[htbp]\n"
        "\\centering\n"
        "\\caption{Deployment evaluation: Early Detection Rate and Time To Detect across all environments.}\n"
        "\\label{tab:eval_deployment}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        f"\\begin{{tabular}}{{{col_spec}}}\n"
        "\\toprule\n"
        f"\\textbf{{Method}} & {env_headers} \\\\\n"
        f"{cmidrules}\n"
        f" & {sub_headers} \\\\\n"
        "\\midrule\n"
    )

    for i, m in enumerate(methods):
        if i in METHOD_GROUP_BREAKS:
            latex += "\\midrule\n"
        cells = [display_methods[i]]
        for env in envs:
            for mkey, _, fmt, _ in metrics:
                v = all_results[env][m][mkey]
                if mkey == 'med_ttd' and not np.isnan(v):
                    v = v / 10.0
                cell = fmt.format(v) if not np.isnan(v) else "---"
                if m == best_per[(env, mkey)]:
                    cell = f"\\textbf{{{cell}}}"
                cells.append(cell)
        latex += " & ".join(cells) + " \\\\\n"

    latex += "\\bottomrule\n\\end{tabular}}%\n\\end{table*}\n"
    return latex


def generate_evaluation_tables(results_dir: Path, env_filter: list[str] | None = None):
    """Generate evaluation LaTeX tables from .npz files."""
    all_results = _load_evaluation_results(results_dir)
    if env_filter:
        all_results = {e: v for e, v in all_results.items() if e in env_filter}
    if not all_results:
        return

    # Filter to methods in DEFAULT_METHODS, in order
    default_keys = [k for k in DEFAULT_METHODS
                    if all(k in all_results[env] for env in all_results)]
    if not default_keys:
        return
    filtered = {
        env: {m: res[m] for m in default_keys}
        for env, res in all_results.items()
    }

    tex = format_evaluation_table(filtered)
    tex_file = results_dir / "deployment.tex"
    tex_file.write_text(tex)
    print(f"  {tex_file}")


# ---------------------------------------------------------------------------
# Table generation from .npz files
# ---------------------------------------------------------------------------

def generate_score_quality_tables(results_dir: Path):
    """Generate AUROC table from score_quality .npz files.

    Expects files at ``results_dir/{env}.npz`` with keys like
    ``auroc_{method_display_key}``.
    """
    npz_files = sorted(results_dir.glob("*.npz"))
    if not npz_files:
        return

    # Build {env: {display_name: {auroc: value}}}
    all_results = {}
    for npz_file in npz_files:
        env_name = npz_file.stem
        data = dict(np.load(npz_file, allow_pickle=True))

        env_results = {}
        for key in data:
            if not key.startswith("auroc_"):
                continue
            method_key = key[len("auroc_"):]
            display = method_key.replace("_", " ")
            env_results[display] = {'auroc': float(data[key])}

        if env_results:
            all_results[env_name] = {'results': env_results, 'stats': {}}

    if not all_results:
        return

    # Filter to default methods present in all envs
    default_display = [get_method_display_name(k) for k in DEFAULT_METHODS]
    default_set = set(default_display)
    eligible = {e: d for e, d in all_results.items()
                if default_set.issubset(d['results'].keys())}

    if eligible:
        filtered = {
            env: {'results': {m: d['results'][m] for m in default_display},
                  'stats': d['stats']}
            for env, d in eligible.items()
        }
        tex = format_metric_latex_table(
            filtered, 'auroc',
            caption='AUROC scores across all environments.',
            label='auroc_all_envs',
        )
        tex_file = results_dir / "auroc.tex"
        tex_file.write_text(tex)
        print(f"  {tex_file}")


def generate_compute_cost_tables(results_dir: Path):
    """Generate compute cost LaTeX tables from .npz files."""
    npz_files = sorted(results_dir.glob("*_compute_cost.npz"))
    if not npz_files:
        return

    for npz_file in npz_files:
        env_name = npz_file.stem.replace("_compute_cost", "")
        data = np.load(npz_file, allow_pickle=True)
        results = data['results'].item()

        tex = format_compute_cost_table(results, env_name)
        tex_file = results_dir / f"{env_name}_compute_cost.tex"
        tex_file.write_text(tex)
        print(f"  {tex_file}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    from utils.cli import add_env_arg, parse_envs

    parser = argparse.ArgumentParser(description="Generate LaTeX tables.")
    add_env_arg(parser)
    args = parser.parse_args()

    env_filter = None
    env_was_explicit = args.env != parser.get_default("env")
    if env_was_explicit:
        env_filter = parse_envs(args)

    results_dir = get_root() / "results"

    print("Generating LaTeX tables...")

    print("\nenv_summary:")
    generate_env_summary_table(results_dir)

    print("\nscore_quality:")
    generate_score_quality_tables(results_dir / "score_quality")

    print("\ndeployment_quality:")
    generate_evaluation_tables(results_dir / "deployment_quality",
                               env_filter=env_filter)

    print("\ncompute_cost:")
    generate_compute_cost_tables(results_dir / "compute_cost")

    print("\nDone.")


if __name__ == "__main__":
    main()
