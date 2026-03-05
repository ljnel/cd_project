
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from config.datasets import DATASETS
from config.detectors import DEFAULT_METHODS, get_method_display_name
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

def format_latex_table(
    results: dict[str, dict[str, tuple[float, float]]],
    env_name: str,
    window: int = None,
    horizon: int = None,
    obs_dim: int = 4,
    train_size: int = 100,
    test_size: int = None,
    failure_prop: float = None
) -> str:
    """Format results as a LaTeX table matching the paper style."""
    display_name = get_env_display_name(env_name)

    # Build stats table header
    stats_table = f"""
\\begin{{table}}[h!]
\\centering
\\begin{{tabular}}{{|l|c|c|c|c|c|c|}}
\\hline
Environment & W & H & Obs dim & Size of train & Size of test & Prop. of failures in test \\\\ \\hline
{display_name} & {window or '?'} & {horizon or '?'} & {obs_dim} & {train_size} & {test_size or '?'} & {failure_prop or '?'} \\\\ \\hline
\\end{{tabular}}
\\caption{{Training and testing data statistics for {display_name}.}}
\\label{{tab:{env_name.lower()}_stats}}
\\end{{table}}
"""

    # Build results table
    results_table = f"""
\\begin{{table}}[h!]
\\centering
\\begin{{tabular}}{{|l|cccc|}}
\\hline
\\multirow{{2}}{{*}}{{Method}}
  & \\multicolumn{{4}}{{c|}}{{{display_name}}} \\\\ \\cline{{2-5}}
 & TNR (\\%) & TPR (\\%) & TPR@5\\%FPR & AUROC \\\\ \\hline
"""

    for method_name, metrics in results.items():
        tnr_mean, tnr_std = metrics['TNR']
        tpr_mean, tpr_std = metrics['TPR']
        tpr5_mean, tpr5_std = metrics['TPR@5%FPR']
        auroc_mean, auroc_std = metrics['AUROC']

        results_table += f"""
{method_name}
 & {tnr_mean:.2f} $\\pm$ {tnr_std:.2f} & {tpr_mean:.2f} $\\pm$ {tpr_std:.2f} & {tpr5_mean:.3f} $\\pm$ {tpr5_std:.3f} & {auroc_mean:.3f} $\\pm$ {auroc_std:.3f}\\\\
"""

    results_table += """
\\hline
\\end{tabular}
\\caption{Results for the """ + display_name + """ environment.}
\\label{tab:""" + env_name.lower() + """}
\\end{table}
"""

    return stats_table + "\n" + results_table


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
    col_spec = "|l|" + "c" * len(envs) + "|c|"
    env_display_names = [get_env_display_name(env) for env in envs]
    header_row = " & ".join(env_display_names)

    latex = f"""
\\begin{{table*}}[h!]
\\centering
\\begin{{tabular}}{{{col_spec}}}
\\hline
Method & {header_row} & Avg. Rank \\\\ \\hline
"""

    # Compute average ranks across environments
    avg_ranks = compute_avg_ranks(all_results, metric_key, methods, envs)
    best_rank_method = min(avg_ranks, key=avg_ranks.get)

    # Add rows for each method
    for method in methods:
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

    latex += f"""\\hline
\\end{{tabular}}
\\caption{{{caption}}}
\\label{{tab:{label}}}
\\end{{table*}}
"""
    return latex


def format_panda_table(
    results: dict[str, dict[str, float]],
    n_train: int | None = None,
    n_test: int | None = None,
    anomaly_prop: float | None = None,
) -> str:
    """Format panda trajectory-level results as a LaTeX table."""
    latex = "\\begin{table}[h!]\n\\centering\n"
    latex += "\\begin{tabular}{|l|cccc|}\n\\hline\n"
    latex += "Method & TNR (\\%) & TPR (\\%) & TPR@5\\%FPR & AUROC \\\\ \\hline\n"

    for name, m in results.items():
        latex += (f"{name} & {m['TNR']:.2f} & {m['TPR']:.2f} "
                  f"& {m['TPR@5%FPR']:.3f} & {m['AUROC']:.3f} \\\\\n")

    latex += "\\hline\n\\end{tabular}\n"
    latex += "\\caption{Trajectory-level anomaly detection on the Panda dataset."
    if n_train is not None and n_test is not None and anomaly_prop is not None:
        latex += (f" Train: {n_train} expert trajectories, "
                  f"Test: {n_test} trajectories "
                  f"(anomaly proportion: {anomaly_prop:.3f}).")
    latex += "}\n\\label{tab:panda_results}\n\\end{table}\n"
    return latex


def format_compute_cost_table(
    results: dict[str, dict[str, float]],
    env_name: str,
) -> str:
    """Format computational cost results as a LaTeX table."""
    display_name = get_env_display_name(env_name)

    latex = "\n\\begin{table}[h!]\n\\centering\n"
    latex += "\\begin{tabular}{|l|c|c|c|}\n\\hline\n"
    latex += "Method & Train (s) & Predict Total (ms) & Predict/Sample (ms) \\\\ \\hline\n"

    for method, m in results.items():
        train = f"{m['train_time_mean']:.2f} $\\pm$ {m['train_time_std']:.2f}"
        pred = f"{m['predict_time_mean']:.1f} $\\pm$ {m['predict_time_std']:.1f}"
        per_sample = (f"{m['predict_per_sample_mean']:.3f} $\\pm$ "
                      f"{m['predict_per_sample_std']:.3f}")
        latex += f"{method} & {train} & {pred} & {per_sample} \\\\\n"

    latex += "\\hline\n\\end{tabular}\n"
    latex += f"\\caption{{Computational cost on {display_name.lower()}.}}\n"
    latex += f"\\label{{tab:{env_name}_compute}}\n"
    latex += "\\end{table}\n"
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
    mass_range, friction_range, damping_range, failure_prop (float or None).
    """
    n_envs = len(env_keys)
    assert len(rows) == n_envs

    latex = "\\begin{table*}[h!]\n\\centering\n"
    latex += "\\begin{tabular}{|l|c|c|c|c|c|c|c|}\n\\hline\n"
    latex += ("Environment & $W$ & $H$ & Obs dim & Mass & Friction "
              "& Damping & Failure prop. \\\\ \\hline\n")

    for row in rows:
        fp = f"{row['failure_prop']:.3f}" if row['failure_prop'] is not None else "---"
        latex += (
            f"{row['display_name']} & {row['W']} & {row['H']} & {row['obs_dim']} "
            f"& {_format_range(row['mass_range'])} "
            f"& {_format_range(row['friction_range'])} "
            f"& {_format_range(row['damping_range'])} "
            f"& {fp} \\\\\n"
        )

    latex += "\\hline\n\\end{tabular}\n"
    latex += "\\caption{Summary of environments and datasets.}\n"
    latex += "\\label{tab:env_summary}\n"
    latex += "\\end{table*}\n"
    return latex



def generate_env_summary_table(output_dir: Path):
    """Generate environment summary LaTeX tables from configs and dataset files."""
    env_keys = list(TASK_CONFIGS.keys())
    rows = []

    for env in env_keys:
        task_cfg = TASK_CONFIGS[env]
        env_info = ENV_INFO[env]
        ds_key = f"{env}/fail_pred"
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
            'obs_dim': env_info.obs_dim,
            'mass_range': mass_range,
            'friction_range': friction_range,
            'damping_range': damping_range,
            'failure_prop': failure_prop,
        })

    output_dir.mkdir(parents=True, exist_ok=True)

    tex = format_env_summary_table(env_keys, rows)
    tex_file = output_dir / "env_summary.tex"
    tex_file.write_text(tex)
    print(f"  {tex_file}")


# ---------------------------------------------------------------------------
# Table generation from .npz files
# ---------------------------------------------------------------------------

def generate_fail_pred_tables(results_dir: Path):
    """Generate all fail_pred LaTeX tables from .npz files."""
    npz_files = sorted(results_dir.glob("*_experiment_results.npz"))
    if not npz_files:
        return

    all_results = {}
    for npz_file in npz_files:
        env_name = npz_file.stem.replace("_experiment_results", "")
        data = np.load(npz_file, allow_pickle=True)
        results = data['results'].item()
        stats = data['stats'].item()

        # Per-env table
        tex = format_latex_table(
            results, env_name,
            window=stats.get('win'),
            horizon=stats.get('hor'),
            obs_dim=stats.get('obs_dim', 4),
            train_size=stats.get('n_successes'),
            test_size=stats.get('eps_per_fold'),
            failure_prop=(f"{stats['failure_prop']:.3f}"
                          if stats.get('failure_prop') else None),
        )
        tex_file = results_dir / f"{env_name}_results.tex"
        tex_file.write_text(tex)
        print(f"  {tex_file}")

        all_results[env_name] = {'results': results, 'stats': stats}

    # Combined tables: use default methods, include envs that have all of them
    default_display = [get_method_display_name(k) for k in DEFAULT_METHODS]
    default_set = set(default_display)
    eligible = {e: d for e, d in all_results.items()
                if default_set.issubset(d['results'].keys())}
    if len(eligible) > 1:
        filtered = {}
        for env, d in eligible.items():
            filtered[env] = {
                'results': {m: d['results'][m] for m in default_display},
                'stats': d['stats'],
            }

        auroc_tex = format_metric_latex_table(
            filtered, 'AUROC',
            caption='AUROC scores across all environments.',
            label='auroc_all_envs',
        )
        tex_file = results_dir / "all_envs_auroc_results.tex"
        tex_file.write_text(auroc_tex)
        print(f"  {tex_file}")

        tpr_tex = format_metric_latex_table(
            filtered, 'TPR@5%FPR',
            caption='TPR@5\\%FPR scores across all environments.',
            label='tpr_all_envs',
        )
        tex_file = results_dir / "all_envs_tpr_results.tex"
        tex_file.write_text(tpr_tex)
        print(f"  {tex_file}")


def generate_panda_tables(results_dir: Path):
    """Generate panda LaTeX tables from .npz files."""
    npz_file = results_dir / "panda_experiment_results.npz"
    if not npz_file.exists():
        return

    data = np.load(npz_file, allow_pickle=True)
    results = data['results'].item()
    n_train = int(data['n_train']) if 'n_train' in data.files else None
    n_test = int(data['n_test']) if 'n_test' in data.files else None
    anomaly_prop = float(data['anomaly_prop']) if 'anomaly_prop' in data.files else None

    tex = format_panda_table(results, n_train, n_test, anomaly_prop)
    tex_file = results_dir / "panda_results.tex"
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
    results_dir = get_root() / "results"

    print("Generating LaTeX tables...")

    print("\nenv_summary:")
    generate_env_summary_table(results_dir)

    print("\nfail_pred:")
    generate_fail_pred_tables(results_dir / "fail_pred")

    print("\npanda:")
    generate_panda_tables(results_dir / "panda")

    print("\ncompute_cost:")
    generate_compute_cost_tables(results_dir / "compute_cost")

    print("\nDone.")


if __name__ == "__main__":
    main()
