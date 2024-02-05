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
    latex_str += "\\hline\n\\end{tabular}\n\\label{tab:phase-model-performance-metrics}\n\\caption{Performance metrics for reinforcement learning and direct control models.}\n\\end{table}\n"

    return latex_str


def generate_combat_metrics_latex_table(dc_ep_combat, rl_ep_combat, dc_overall_combat, rl_overall_combat):
    """
    Generate a LaTeX table for combat metrics data from the given dataframes.

    Args:
    - dc_ep_combat: DataFrame containing Direct Control episode combat metrics
    - rl_ep_combat: DataFrame containing Reinforcement Learning episode combat metrics
    - dc_overall_combat: DataFrame containing Direct Control overall combat metrics
    - rl_overall_combat: DataFrame containing Reinforcement Learning overall combat metrics

    Returns:
    - A string containing the LaTeX code for the table
    """
    metrics = {
        'mean_total_reward': 'total reward',
        'mean_lw_exploded': 'lw exploded',
        'mean_lm_shooted': 'lm shooted',
        'mean_lm_crashes': 'lm crashes',
        'mean_lm_ally_collisions': 'ally collisions',
        'mean_lm_exploded_per_ally': 'exploded by ally',
        'mean_lm_out_of_bounds': 'lm out of bounds',
    }

    # Begin the LaTeX table with a smaller font size
    latex_str = "\\begin{table}[H]\n\\centering\n\\begin{tabular}{|l|lll|l|lll|l|}\n\\hline\n"

    # Header for the table
    header = [
        "\\textbf{Scenario}", "\\textbf{5x5}", "\\textbf{10x5}", "\\textbf{15x5}", "\\textbf{All RL}",
        "\\textbf{5x5}", "\\textbf{10x5}", "\\textbf{15x5}", "\\textbf{All DC}"
    ]
    latex_str += " & ".join(header) + " \\\\\n\\hline\n"

    # Extract the metric names by excluding the first column for the scenario names
    #metrics = dc_ep_combat.columns.tolist()[1:]

    # Create rows for each metric
    for metric, label in metrics.items():
        # Properly format metric names for LaTeX
        row = [label] # metric_label = metric.replace('_', ' ') # metric.replace('_', '\\_')
        # Start the row with the metric label
        #row = [metric_label]
        # Add RL episode metrics
        for _, row_values in rl_ep_combat.iterrows():
            row.append(f"{row_values[metric]:.1f}")
        # Add RL overall metrics
        row.append(f"{rl_overall_combat[metric].values[0]:.1f}")
        # Add DC episode metrics
        for _, row_values in dc_ep_combat.iterrows():
            row.append(f"{row_values[metric]:.1f}")
        # Add DC overall metrics
        row.append(f"{dc_overall_combat[metric].values[0]:.1f}")
        # Join the row values and add to the LaTeX string
        latex_str += " & ".join(map(str, row)) + " \\\\\n"

    # Close the LaTeX table
    latex_str += "\\hline\n\\end{tabular}\n\\label{tab:phase-model-combat-metrics}\n\\caption{Combat metrics for Reinforcement Learning and Direct Control models.}\n\\end{table}\n"

    return latex_str

# Function to reorder the rows in the DataFrame according to the specified scenario order
def reorder_dataframe(df, scenario_order):
    df['scenario'] = pd.Categorical(df['scenario'], categories=scenario_order, ordered=True)
    return df.sort_values('scenario')

# -------------------------------------------------------------------------------------------------------------------------

scenario_order = ["5x5", "10x5", "15x5"]

dc_ep_path = 'apps/models/ma_quadx_chaser_20240202-014543/ep_data/ma_quadx_chaser-30000000/dc/dc-ep_performance_metrics.csv'
rl_ep_path = 'apps/models/ma_quadx_chaser_20240202-014543/ep_data/ma_quadx_chaser-30000000/rl/rl-ep_performance_metrics.csv'
dc_overall_path = 'apps/models/ma_quadx_chaser_20240202-014543/ep_data/ma_quadx_chaser-30000000/dc/dc-overall_performance_metrics.csv'
rl_overall_path = 'apps/models/ma_quadx_chaser_20240202-014543/ep_data/ma_quadx_chaser-30000000/rl/rl-ep_performance_metrics.csv'


dc_ep_df = reorder_dataframe(pd.read_csv(dc_ep_path), scenario_order)
rl_ep_df = reorder_dataframe(pd.read_csv(rl_ep_path), scenario_order)
dc_overall_df = reorder_dataframe(pd.read_csv(dc_overall_path), scenario_order)
rl_overall_df = reorder_dataframe(pd.read_csv(rl_overall_path), scenario_order)

# Generate the LaTeX code using the dataframes
latex_code = generate_compact_latex_table(dc_ep_df, rl_ep_df, dc_overall_df, rl_overall_df)

# Since the latex code is quite long, let's save it to a text file instead of printing it out
latex_file_path = Path('apps/models/ma_quadx_chaser_20240202-014543/latex/performance_metrics_table.txt')
latex_file_path.parent.mkdir(parents=True, exist_ok=True)
with open(latex_file_path, 'w') as file:
    file.write(latex_code)
print(f'Performance metrics latex table saved at: {latex_file_path}')

# ******* Combat Metrics ************

dc_ep_combat_path = 'apps/models/ma_quadx_chaser_20240202-014543/ep_data/ma_quadx_chaser-30000000/dc/dc-ep_combat_metrics.csv'
rl_ep_combat_path = 'apps/models/ma_quadx_chaser_20240202-014543/ep_data/ma_quadx_chaser-30000000/rl/rl-ep_combat_metrics.csv'
dc_overall_combat_path = 'apps/models/ma_quadx_chaser_20240202-014543/ep_data/ma_quadx_chaser-30000000/dc/dc-overall_combat_metrics.csv'
rl_overall_combat_path = 'apps/models/ma_quadx_chaser_20240202-014543/ep_data/ma_quadx_chaser-30000000/rl/rl-ep_combat_metrics.csv'

# Read the new CSV files into dataframes
dc_ep_combat_df = reorder_dataframe(pd.read_csv(dc_ep_combat_path), scenario_order)
rl_ep_combat_df = reorder_dataframe(pd.read_csv(rl_ep_combat_path), scenario_order)
dc_overall_combat_df = reorder_dataframe(pd.read_csv(dc_overall_combat_path), scenario_order)
rl_overall_combat_df = reorder_dataframe(pd.read_csv(rl_overall_combat_path), scenario_order)

combat_metrics_latex_code = generate_combat_metrics_latex_table(dc_ep_combat_df, rl_ep_combat_df, dc_overall_combat_df, rl_overall_combat_df)

# Save the LaTeX code to a text file
combat_metrics_latex_file_path = 'apps/models/ma_quadx_chaser_20240202-014543/latex/combat_metrics_table.txt'
with open(combat_metrics_latex_file_path, 'w') as file:
    file.write(combat_metrics_latex_code)

print(f'Combat metrics latex table saved at: {combat_metrics_latex_file_path}')

