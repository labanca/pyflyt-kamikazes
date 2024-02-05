from pathlib import Path
from pprint import pprint

import pandas as pd
import glob
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib

from modules.utils import save_dict_to_csv

matplotlib.use('TkAgg')

def append_csvs(experiment_name, model_name, base_path="/app/models", eval_mode='rl'):
    """
    Append all CSVs from a given experiment and model into a single DataFrame.

    Parameters:
    experiment_name (str): The name of the experiment.
    model_name (str): The name of the model.
    base_path (str): The base path where the CSV files are stored.

    Returns:
    DataFrame: A concatenated DataFrame containing all episodes data.
    """
    # Construct the path pattern for the CSV files
    path_pattern = f"{base_path}/{experiment_name}/ep_data/{model_name}/{eval_mode}/*x*.csv"

    # Use glob to match all csv files following the pattern
    all_files = glob.glob(path_pattern)

    # List to hold dataframes
    df_list = []

    # Iterate through the list of files and read each one into a DataFrame
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        df_list.append(df)

    # Concatenate all dataframes into a single one
    concatenated_df = pd.concat(df_list, axis=0, ignore_index=True)

    output_file_path = f"{base_path}/{experiment_name}/ep_data/{model_name}/{eval_mode}/consolidate_scenarios.csv"
    concatenated_df.to_csv(output_file_path, index=False)


def calculate_metrics(input_file_path, output_file_path):
    """
    Calculate metrics for each scenario and for all scenarios together from the consolidated CSV file.

    Parameters:
    input_file_path (str): The path to the input CSV file containing all episodes data.
    output_file_path (str): The path where the metrics CSV file will be saved.

    Returns:
    str: The path to the saved CSV file containing the calculated metrics.
    """
    # Read the consolidated scenarios CSV file into a DataFrame
    df = pd.read_csv(input_file_path)

    # Define the calculation for each metric
    def calc_win_rate(group):
        return 100.0 * (group['mission_complete'].sum()/ float(len(group)))

    def calc_explosion_rate(group):
        return 100.0 * group['exploded_target'].sum()/ float(group['num_lm'].sum())

    def calc_destroyed_lm_rate(group):
        return 100.0 * group['downed'].sum()/ float(group['num_lm'].sum())

    def calc_survival_rate(group):
        return 100.0 * group['survived'].sum()/ float(group['num_lm'].sum())

    def calc_casualty_rate(group):
        return 100.0 * (group['crashes'] + group['ally_collision']
                      + group['out_of_bounds'] + group['exploded_by_ally']
                      + group['downed']).sum()/ float(group['num_lm'].sum())

    def calc_timeover_rate(group):
        return 100.0 * group['timeover'].sum() / float(group['num_lm'].sum())

    def calc_time_overs(group):
        return 100.0 * (
                group[group['mission_complete'] == False]['timeover'].sum()
                /(len(group[group['mission_complete'] == False])
                  if len(group[group['mission_complete'] == False]) else 1.0
                  )
        )

    def calc_total_rate(group):
        return (
            calc_explosion_rate(group) +
            calc_survival_rate(group) +
            calc_casualty_rate(group) +
            calc_timeover_rate(group)
        )
    def calc_mean_total_reward(group):
        return group['agents_total_acc_rewards'].mean()

    def calc_mean_lm_shooted(group):
        return group['downed'].mean()

    def calc_mean_lw_exploded(group):
        return group['exploded_target'].mean()

    def calc_mean_lm_crashes(group):
        return group['crashes'].mean()

    def calc_mean_lm_ally_collisions(group):
        return group['ally_collision'].mean()

    def calc_mean_lm_exploded_by_ally(group):
        return group['exploded_by_ally'].mean()

    def calc_mean_lm_out_of_bounds(group):
        return group['out_of_bounds'].mean()

    def calc_mean_lm_timeovers(group):
        return group['timeover'].mean()

    def calc_mean_time_to_complete(group):
        return group[group['mission_complete'] == True]['elapsed_time'].mean()


    # Apply calculations for each scenario
    ep_performance_metric = df.groupby('scenario').apply(lambda group: pd.Series({
        'win_rate': calc_win_rate(group),
        'explosion_rate': calc_explosion_rate(group),
        'survival_rate': calc_survival_rate(group),
        'casualty_rate': calc_casualty_rate(group),
        'time_overs_rate': calc_timeover_rate(group),
        #'total': calc_total_rate(group),
        'mean_time_to_complete': calc_mean_time_to_complete(group),


    })).reset_index()

    # Apply calculations for all data
    overall_performance_metrics = pd.DataFrame([{
        'scenario': f'All {eval_mode} scenarios',
        'win_rate': calc_win_rate(df),
        'explosion_rate': calc_explosion_rate(df),
        'survival_rate': calc_survival_rate(df),
        'casualty_rate': calc_casualty_rate(df),
        'time_overs_rate': calc_timeover_rate(df),
        #'total': calc_total_rate(df),,
        'mean_time_to_complete': calc_mean_time_to_complete(df),
    }])

    ep_combat_metrics = df.groupby('scenario').apply(lambda group: pd.Series({
        'mean_total_reward': calc_mean_total_reward(group),
        'mean_lw_exploded': calc_mean_lw_exploded(group),
        'mean_lm_shooted': calc_mean_lm_shooted(group),
        'mean_lm_crashes': calc_mean_lm_crashes(group),
        'mean_lm_ally_collisions': calc_mean_lm_ally_collisions(group),
        'mean_lm_exploded_per_ally': calc_mean_lm_exploded_by_ally(group),
        'mean_lm_out_of_bounds': calc_mean_lm_out_of_bounds(group),
        'mean_lm_timeovers': calc_mean_lm_timeovers(group),
    })).reset_index()

    overall_combat_metrics = pd.DataFrame([{
        'scenario': f'All {eval_mode} scenarios',
        'mean_total_reward': calc_mean_total_reward(df),
        'mean_lw_exploded': calc_mean_lw_exploded(df),
        'mean_lm_shooted': calc_mean_lm_shooted(df),
        'mean_lm_crashes': calc_mean_lm_crashes(df),
        'mean_lm_ally_collisions': calc_mean_lm_ally_collisions(df),
        'mean_lm_exploded_per_ally': calc_mean_lm_exploded_by_ally(df),
        'mean_lm_out_of_bounds': calc_mean_lm_out_of_bounds(df),
        'mean_lm_timeovers': calc_mean_lm_timeovers(df)
    }])

    #per = pd.DataFrame(overall_metrics, columns=overall_metrics.keys().values)

    ep_performance_metric.to_csv(f"{output_file_path}/{eval_mode}-ep_performance_metrics.csv", index=False)
    ep_combat_metrics.to_csv(f"{output_file_path}/{eval_mode}-ep_combat_metrics.csv", index=False)

    overall_performance_metrics.to_csv(f"{output_file_path}/{eval_mode}-overall_performance_metrics.csv", index=False)
    overall_combat_metrics.to_csv(f"{output_file_path}/{eval_mode}-overall_combat_metrics.csv", index=False)

    # Save the metrics to a CSV file


    return output_file_path



# --------------------------------------------------------------------------------------------

def evaluate_episodes(data, metric_columns, scenario_column='scenario'):
    """
    Evaluate episodes by calculating descriptive statistics, confidence intervals, and plotting metrics for convergence.

    Args:
    - data (DataFrame): The dataset containing episodes and their metrics.
    - metric_columns (list): List of columns in the data that are metrics to be analyzed.
    - scenario_column (str): The name of the column in data that identifies the scenario of each episode.

    Returns:
    - analysis_results (dict): A dictionary containing the analysis results for each scenario.
    """
    analysis_results = {}

    for scenario in data[scenario_column].unique():
        scenario_data = data[data[scenario_column] == scenario]
        scenario_analysis = {}

        for metric in metric_columns:
            metric_data = scenario_data[metric]

            # Descriptive statistics
            mean = metric_data.mean()
            median = metric_data.median()
            std_dev = metric_data.std()
            min_val = metric_data.min()
            max_val = metric_data.max()

            # Confidence Intervals (95%)
            ci_low, ci_high = stats.norm.interval(0.95, loc=mean, scale=std_dev/np.sqrt(len(metric_data)))

            # Save the analysis in the dictionary
            scenario_analysis[metric] = {
                'mean': mean,
                'median': median,
                'std_dev': std_dev,
                'min': min_val,
                'max': max_val,
                '95%_ci_low': ci_low,
                '95%_ci_high': ci_high
            }

            # Convergence Analysis Plot
            plt.figure(figsize=(10, 6))
            plt.plot(metric_data.index, metric_data, label=metric)
            plt.title(f'Convergence Plot for {metric} in Scenario {scenario}')
            plt.xlabel('Episode')
            plt.ylabel(metric)
            plt.legend()
            plt.grid(True)
            plt.show()

        analysis_results[scenario] = scenario_analysis

    return analysis_results



eval_mode = 'rl'
base_path = 'apps/models'
experiment_name = 'ma_quadx_chaser_20240204-120343'
model_name = 'ma_quadx_chaser-20027008'

#append_csvs(experiment_name=experiment_name, model_name=model_name, base_path=base_path, eval_mode=eval_mode)

input_path = f"{base_path}/{experiment_name}/ep_data/{model_name}/{eval_mode}/consolidate_scenarios.csv"
metrics_path = f"{base_path}/{experiment_name}/ep_data/{model_name}/{eval_mode}"


#calculate_metrics(input_file_path=input_path, output_file_path=metrics_path)

# Assuming 'data' is your dataframe and it includes a column 'scenario'
# and the metrics you're interested in are in columns like 'metric1', 'metric2', ...
# You would call the function like this:
metric_columns = ['exploded_target']  # replace with your actual metric column names

consolidate_scenarios_path = 'apps/models/ma_quadx_chaser_20240204-120343/ep_data/ma_quadx_chaser-20027008/rl/consolidate_scenarios.csv'
data = pd.read_csv(consolidate_scenarios_path)

# Call the function with your data
results = evaluate_episodes(data, metric_columns, 'scenario')

# The results will be a dictionary where each key is a scenario, and the values are the analyses for that scenario
pprint(results)
confidence_interval_path = 'apps/models/ma_quadx_chaser_20240204-120343/ep_data/ma_quadx_chaser-20027008/rl/confidence_interval'
save_dict_to_csv(results, confidence_interval_path)


