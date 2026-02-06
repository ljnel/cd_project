from typing import Dict, Tuple

from config.envs import ENV_INFO


def get_env_display_name(env_key: str) -> str:
    """Get display name for an environment from config."""
    info = ENV_INFO.get(env_key)
    return info.display_name if info else env_key


def format_latex_table(
    results: Dict[str, Dict[str, Tuple[float, float]]],
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
 & TNR (\\%) & TPR (\\%) & F2 & AUROC \\\\ \\hline
"""

    for method_name, metrics in results.items():
        tnr_mean, tnr_std = metrics['TNR']
        tpr_mean, tpr_std = metrics['TPR']
        f2_mean, f2_std = metrics['F2']
        auroc_mean, auroc_std = metrics['AUROC']

        results_table += f"""
{method_name}
 & {tnr_mean:.2f} $\\pm$ {tnr_std:.2f} & {tpr_mean:.2f} $\\pm$ {tpr_std:.2f} & {f2_mean:.3f} $\\pm$ {f2_std:.3f} & {auroc_mean:.3f} $\\pm$ {auroc_std:.3f}\\\\
"""

    results_table += """
\\hline
\\end{tabular}
\\caption{Results for the """ + display_name + """ environment.}
\\label{tab:""" + env_name.lower() + """}
\\end{table}
"""

    return stats_table + "\n" + results_table


def format_f2_latex_table(
    all_results: Dict[str, Dict[str, Dict[str, Tuple[float, float]]]]
) -> str:
    """
    Format F2 scores as a single LaTeX table with environments as columns.
    The method with the highest mean F2 score for each environment is bolded.
    """
    envs = list(all_results.keys())
    methods = list(next(iter(all_results.values()))['results'].keys())

    # Find best method for each environment
    best_method_per_env = {}
    for env in envs:
        best_mean = -1
        best_method = None
        for method in methods:
            f2_mean, _ = all_results[env]['results'][method]['F2']
            if f2_mean > best_mean:
                best_mean = f2_mean
                best_method = method
        best_method_per_env[env] = best_method

    # Build table header
    col_spec = "|l|" + "c|" * len(envs)
    env_display_names = [get_env_display_name(env) for env in envs]
    header_row = " & ".join(env_display_names)

    latex = f"""
\\begin{{table}}[h!]
\\centering
\\begin{{tabular}}{{{col_spec}}}
\\hline
Method & {header_row} \\\\ \\hline
"""

    # Add rows for each method
    for method in methods:
        cells = [method]
        for env in envs:
            f2_mean, f2_std = all_results[env]['results'][method]['F2']
            cell = f"{f2_mean:.3f} $\\pm$ {f2_std:.3f}"
            if method == best_method_per_env[env]:
                cell = f"\\textbf{{{cell}}}"
            cells.append(cell)
        latex += " & ".join(cells) + " \\\\\n"

    latex += """\\hline
\\end{tabular}
\\caption{F2 scores across all environments.}
\\label{tab:f2_all_envs}
\\end{table}
"""
    return latex
