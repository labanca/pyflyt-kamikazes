from pathlib import Path

import pandas as pd

# Redefine the function to generate the compact LaTeX table
def generate_performance_latex_table(dc_ep, rl_ep, dc_overall, rl_overall):
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
    latex_str += "\\hline\n\\end{tabular}\n" \
                 "\\label{tab:complete-model-performance-metrics}\n" \
                 "\\caption{Performance metrics for reinforcement learning and direct control models. The win, " \
                 "explosion, survival, casualty, and time over rates metrics are related to all 500 episodes of the " \
                 "corresponding scenario. The mean time metric is measured in seconds and also related to all episodes " \
                 "of the corresponding scenario. The 2nd, 3rd, and 4th columns are the scenarios with 5 LM and 5 LW," \
                 " 10 LM and 5 LW, and 15 LM and 5 LW, respectively. The 6th, 7th, and 8th columns are the DC scenarios " \
                 "with the same LM/LW distribution as their RL counterparts. The 5th and 9th columns comprehend the total " \
                 "of the three RL and DC scenarios, respectively.}\n" \
                 "\\end{table}\n"

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
        'mean_total_reward': 'Mean reward',
        'mean_lw_exploded': 'Exploded target',
        'mean_lm_shooted': 'Shooted',
        'mean_lm_survived': 'Survived',
        'mean_lm_crashes': 'Crashes',
        'mean_lm_ally_collisions': 'Ally collisions',
        'mean_lm_exploded_per_ally': 'Exploded by ally',
        'mean_lm_out_of_bounds': 'Out of bounds',
        'mean_lm_timeovers': 'timeovers',
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


# Define the LaTeX table generation function
def generate_mean_std_latex_table(rl_df, dc_df, scenarios, metrics_to_use):
    """
    Generate LaTeX table with mean and standard deviation for RL and DC scenarios for specified metrics,
    with scenario column lines merged and a horizontal line after the last metric of each scenario.

    Args:
    - rl_df: DataFrame for RL scenarios with metrics and their values.
    - dc_df: DataFrame for DC scenarios with metrics and their values.
    - scenarios: List of scenarios to include in the table.
    - metrics_to_use: List of metric names to include in the table.

    Returns:
    - A string containing the LaTeX code for the table.
    """
    metrics_mapping = {
        'agents_mean_acc_rewards': 'Mean reward',
        'exploded_target': 'Exploded target',
        'downed': 'Shooted',
        'survived': 'Survived',
        'crashes': 'Crashes',
        'ally_collision': 'Ally collisions',
        'exploded_by_ally': 'Exploded by ally',
        'out_of_bounds': 'Out of bounds',
        'timeover': 'Timeovers',
    }

    # Start the LaTeX table
    latex_str = "\\begin{table}[H]\n\\centering\n\\begin{tabular}{|c|l|cc|cc|}\n\\hline\n"
    latex_str += "\\multirow{1}*{\\textbf{Scenario}} & \\textbf{Metric} & \\multicolumn{2}{c|}{\\textbf{RL Scenarios}} & \\multicolumn{2}{c|}{\\textbf{DC Scenarios}} \\\\\n"
    latex_str += " & & mean & std dev & mean & std dev \\\\\n\\hline\n"

    # Iterate over scenarios and specified metrics
    for scenario in scenarios:

        if scenario == "All Scenarios":
            rl_scenario_df = rl_df[rl_df['scenario'].isin(['5x5', '10x5', '15x5'])]
            dc_scenario_df = dc_df[dc_df['scenario'].isin(['5x5', '10x5', '15x5'])]
        else:
            rl_scenario_df = rl_df[rl_df['scenario'] == scenario]
            dc_scenario_df = dc_df[dc_df['scenario'] == scenario]
            metric_count = len(metrics_to_use)

        for i, metric in enumerate(metrics_to_use):
            rl_mean = rl_scenario_df[metric].mean()
            rl_std = rl_scenario_df[metric].std()
            dc_mean = dc_scenario_df[metric].mean()
            dc_std = dc_scenario_df[metric].std()

            # Add row to LaTeX table
            if i == 0:
                # Merge scenario column lines for the first metric of the scenario
                latex_str += f"\\multirow{{{metric_count}}}{{*}}{{{scenario}}} & {metrics_mapping[metric]} & {rl_mean:.2f} & {rl_std:.2f} & {dc_mean:.2f} & {dc_std:.2f} \\\\\n"
            else:
                latex_str += f" & {metrics_mapping[metric]} & {rl_mean:.2f} & {rl_std:.2f} & {dc_mean:.2f} & {dc_std:.2f} \\\\\n"

        # Add a horizontal line after the last metric for the scenario
        latex_str += "\\hline\n"

    # Close the LaTeX table
    latex_str += "\\end{tabular}\n" \
                 "\\caption{Mean values and standard deviation of the collect combat metrics.}" \
                 "\\label{tab:complete-model-combat-metrics}"\
                 "\n\\end{table}\n"

    return latex_str

# -------------------------------------------------------------------------------------------------------------------------

scenario_order = ["5x5", "10x5", "15x5"]
base_path = 'apps/models'
experiment_name = 'ma_quadx_chaser_20240204-120343'
model_name = 'model_39500000'
eval_mode = ['dc', 'rl']

input_path = f"{base_path}/{experiment_name}/ep_data/{model_name}"


# ******* Performance Metrics ************
dc_ep_path = f'{input_path}/dc/dc-ep_performance_metrics.csv'
rl_ep_path = f'{input_path}/rl/rl-ep_performance_metrics.csv'
dc_overall_path = f'{input_path}/dc/dc-overall_performance_metrics.csv'
rl_overall_path = f'{input_path}/rl/rl-overall_performance_metrics.csv'

dc_ep_df = reorder_dataframe(pd.read_csv(dc_ep_path), scenario_order)
rl_ep_df = reorder_dataframe(pd.read_csv(rl_ep_path), scenario_order)
dc_overall_df = reorder_dataframe(pd.read_csv(dc_overall_path), scenario_order)
rl_overall_df = reorder_dataframe(pd.read_csv(rl_overall_path), scenario_order)

# Generate the LaTeX code using the dataframes
latex_code = generate_performance_latex_table(dc_ep_df, rl_ep_df, dc_overall_df, rl_overall_df)

# Save latex performance
latex_file_path = Path(f"{base_path}/{experiment_name}/latex/performance_metrics_table.txt")
latex_file_path.parent.mkdir(parents=True, exist_ok=True)
with open(latex_file_path, 'w') as file:
    file.write(latex_code)
print(f'Performance metrics latex table saved at: {latex_file_path}')

# ******* Combat Metrics ************

dc_ep_combat_path = f'{input_path}/dc/dc-ep_combat_metrics.csv'
rl_ep_combat_path = f'{input_path}/rl/rl-ep_combat_metrics.csv'
dc_overall_combat_path = f'{input_path}/dc/dc-overall_combat_metrics.csv'
rl_overall_combat_path = f'{input_path}/rl/rl-overall_combat_metrics.csv'

# Read the new CSV files into dataframes
dc_ep_combat_df = reorder_dataframe(pd.read_csv(dc_ep_combat_path), scenario_order)
rl_ep_combat_df = reorder_dataframe(pd.read_csv(rl_ep_combat_path), scenario_order)
dc_overall_combat_df = reorder_dataframe(pd.read_csv(dc_overall_combat_path), scenario_order)
rl_overall_combat_df = reorder_dataframe(pd.read_csv(rl_overall_combat_path), scenario_order)

combat_metrics_latex_code = generate_combat_metrics_latex_table(dc_ep_combat_df, rl_ep_combat_df, dc_overall_combat_df, rl_overall_combat_df)

# Save the LaTeX combat metrics table
combat_metrics_latex_file_path = f'{base_path}/{experiment_name}/latex/combat_metrics_table.txt'
with open(combat_metrics_latex_file_path, 'w') as file:
    file.write(combat_metrics_latex_code)
print(f'Combat metrics latex table saved at: {combat_metrics_latex_file_path}')


rl_scenario_consolidated_path = f'{input_path}/rl/rl-consolidate_scenarios.csv'
dc_scenario_consolidated_path = f'{input_path}/dc/dc-consolidate_scenarios.csv'
scenario_order = ['5x5', '10x5', '15x5', 'All Scenarios']
metrics_to_use = ['agents_mean_acc_rewards', 'exploded_target','downed','survived','crashes',
                  'ally_collision','exploded_by_ally','out_of_bounds','timeover']

rl_consolidated_df = pd.read_csv(rl_scenario_consolidated_path)
dc_consolidated_df = pd.read_csv(dc_scenario_consolidated_path)


meanstd_latex_latex_code = generate_mean_std_latex_table(rl_consolidated_df, dc_consolidated_df,
                                                         scenarios=scenario_order, metrics_to_use=metrics_to_use)

meanstd_latex_file_path = f'{base_path}/{experiment_name}/latex/mean_std_scenarios_table.txt'
with open(meanstd_latex_file_path, 'w') as file:
    file.write(meanstd_latex_latex_code)
print(f'mean std latex table saved at: {meanstd_latex_file_path}')
