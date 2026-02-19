
import numpy as np
from scipy.stats import rankdata

from config.envs import ENV_INFO


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
    scores = np.array([
        [all_results[env]['results'][method][metric_key] for method in methods]
        for env in envs
    ])
    # Rank per env (higher metric = rank 1), then average across envs
    ranks = np.array([rankdata(-row, method='average') for row in scores])
    avg_ranks = ranks.mean(axis=0)
    return {method: avg_ranks[i] for i, method in enumerate(methods)}


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
        tpr5 = metrics['TPR@5%FPR']
        auroc = metrics['AUROC']

        results_table += f"""
{method_name}
 & {tnr_mean:.2f} $\\pm$ {tnr_std:.2f} & {tpr_mean:.2f} $\\pm$ {tpr_std:.2f} & {tpr5:.3f} & {auroc:.3f}\\\\
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

    # Find best method for each environment
    best_method_per_env = {}
    for env in envs:
        best_val = -1
        best_method = None
        for method in methods:
            val = all_results[env]['results'][method][metric_key]
            if val > best_val:
                best_val = val
                best_method = method
        best_method_per_env[env] = best_method

    # Build table header
    col_spec = "|l|" + "c" * len(envs) + "|c|"
    env_display_names = [get_env_display_name(env) for env in envs]
    header_row = " & ".join(env_display_names)

    latex = f"""
\\begin{{table}}[h!]
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
            val = all_results[env]['results'][method][metric_key]
            cell = f"{val:.3f}"
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
\\end{{table}}
"""
    return latex
