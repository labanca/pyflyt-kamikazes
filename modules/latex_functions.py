from pathlib import Path

import pandas as pd

# Redefine the function to generate the compact LaTeX table
def generate_compact_latex_table(dc_ep, rl_ep, dc_overall, rl_overall):
    """
    Generate a more compact LaTeX table with reduced font size and adjusted column headers to fit in a single page.

    Args:
    - dc_ep: DataFrame containing Direct Control episode metrics
    - rl_ep: DataFrame containing Reinforcement Learning episode metrics
    - dc_overall: DataFrame containing Direct Control overall metrics
    - rl_overall: DataFrame containing Reinforcement Learning overall metrics

    Returns:
    - A string containing the LaTeX code for the table
    """

    # Define the header for the table with reduced column headers
    header = [
        "\\textbf{Scenario}", "\\textbf{5x5}", "\\textbf{10x5}", "\\textbf{15x5}", "\\textbf{All RL}",
        "\\textbf{5x5}", "\\textbf{10x5}", "\\textbf{15x5}", "\\textbf{All DC}"
    ]

    # Define the metrics to be included, with modified labels to match the provided table
    metrics = {
        'win_rate': 'win rate [\%]',
        'explosion_rate': 'explosion rate [\%]',
        'survival_rate': 'survival rate [\%]',
        'casualty_rate': 'casualty rate [\%]',
        'time_overs_rate': 'timeover rate [\%]',
        'mean_time_to_complete': 'mean time [s]'
    }

    # Begin the LaTeX table with a smaller font size
    latex_str = "\\begin{table}[H]\n\\centering\n\\begin{tabular}{|l|lll|l|lll|l|}\n\\hline\n"

    # Add the header to the LaTeX string
    latex_str += " & ".join(header) + " \\\\\n\\hline\n"

    # Add the data rows
    for metric, label in metrics.items():
        row = [label]
        # RL episodes
        for scenario in rl_ep['scenario']:
            row.append(f"{rl_ep.loc[rl_ep['scenario'] == scenario, metric].values[0]:.1f}")
        # RL overall
        row.append(f"{rl_overall[metric].values[0]:.1f}")
        # DC episodes
        for scenario in dc_ep['scenario']:
            row.append(f"{dc_ep.loc[dc_ep['scenario'] == scenario, metric].values[0]:.1f}")
        # DC overall
        row.append(f"{dc_overall[metric].values[0]:.1f}")
        # Add the row to the LaTeX string
        latex_str += " & ".join(row) + " \\\\\n"

    # Close the LaTeX table
    latex_str += "\\hline\n\\end{tabular}\n\\caption{Compact performance metrics for Reinforcement Learning and Direct Control models.}\n\\end{table}\n"

    return latex_str

dc_ep_df = pd.read_csv('apps/models/ma_quadx_chaser_20240202-014543/ep_data/ma_quadx_chaser-30000000/dc/dc-ep_performance_metric.csv')
rl_ep_df = pd.read_csv('apps/models/ma_quadx_chaser_20240202-014543/ep_data/ma_quadx_chaser-30000000/rl/rl-ep_performance_metric.csv')
dc_overall_df = pd.read_csv('apps/models/ma_quadx_chaser_20240202-014543/ep_data/ma_quadx_chaser-30000000/dc/dc-overall_performance_metrics.csv')
rl_overall_df = pd.read_csv('apps/models/ma_quadx_chaser_20240202-014543/ep_data/ma_quadx_chaser-30000000/rl/rl-ep_performance_metric.csv')

# Generate the LaTeX code using the dataframes
latex_code = generate_compact_latex_table(dc_ep_df, rl_ep_df, dc_overall_df, rl_overall_df)

# Since the latex code is quite long, let's save it to a text file instead of printing it out
latex_file_path = Path('apps/models/ma_quadx_chaser_20240202-014543/latex/performance_metrics_table.txt')
latex_file_path.parent.mkdir(parents=True, exist_ok=True)
with open(latex_file_path, 'w') as file:
    file.write(latex_code)


