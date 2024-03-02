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

    output_file_path = f"{base_path}/{experiment_name}/ep_data/{model_name}/{eval_mode}/{eval_mode}-consolidate_scenarios.csv"
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
        return group['max_achieved_speed'].mean()

    def calc_mean_speed_magintude(group):
        return group['agents_total_acc_rewards'].mean()

    def calc_mean_lm_shooted(group):
        return group['downed'].mean()

    def calc_mean_lw_exploded(group):
        return group['exploded_target'].mean()

    def calc_mean_lm_crashes(group):
        return group['crashes'].mean()

    def calc_mean_lm_survived(group):
        return group['survived'].mean()

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
        'mean_lm_survived': calc_mean_lm_survived(group),
        'mean_lm_crashes': calc_mean_lm_crashes(group),
        'mean_lm_ally_collisions': calc_mean_lm_ally_collisions(group),
        'mean_lm_exploded_per_ally': calc_mean_lm_exploded_by_ally(group),
        'mean_lm_out_of_bounds': calc_mean_lm_out_of_bounds(group),
        'mean_lm_timeovers': calc_mean_lm_timeovers(group),
        'mean_lm_maxspeed': calc_mean_lm_timeovers(group),
    })).reset_index()

    overall_combat_metrics = pd.DataFrame([{
        'scenario': f'All {eval_mode} scenarios',
        'mean_total_reward': calc_mean_total_reward(df),
        'mean_lw_exploded': calc_mean_lw_exploded(df),
        'mean_lm_shooted': calc_mean_lm_shooted(df),
        'mean_lm_survived': calc_mean_lm_survived(df),
        'mean_lm_crashes': calc_mean_lm_crashes(df),
        'mean_lm_ally_collisions': calc_mean_lm_ally_collisions(df),
        'mean_lm_exploded_per_ally': calc_mean_lm_exploded_by_ally(df),
        'mean_lm_out_of_bounds': calc_mean_lm_out_of_bounds(df),
        'mean_lm_timeovers': calc_mean_lm_timeovers(df),
        'mean_lm_maxspeed': calc_mean_lm_timeovers(df),
    }])

    #per = pd.DataFrame(overall_metrics, columns=overall_metrics.keys().values)

    ep_performance_metric.to_csv(f"{output_file_path}/{eval_mode}-ep_performance_metrics.csv", index=False)
    ep_combat_metrics.to_csv(f"{output_file_path}/{eval_mode}-ep_combat_metrics.csv", index=False)

    overall_performance_metrics.to_csv(f"{output_file_path}/{eval_mode}-overall_performance_metrics.csv", index=False)
    overall_combat_metrics.to_csv(f"{output_file_path}/{eval_mode}-overall_combat_metrics.csv", index=False)

    # Save the metrics to a CSV file


    return output_file_path

# --------------------------------------------------------------------------------------------


def scenario_metrics_statistics(data, metric_columns, scenario_column='scenario',
                                   output_file='scenario_analysis.csv'):
    """
    Evaluate episodes by calculating descriptive statistics, plotting metrics for convergence,
    and saving the analysis results for each scenario into a CSV file.

    Args:
    - data (DataFrame): The dataset containing episodes and their metrics.
    - metric_columns (list): List of columns in the data that are metrics to be analyzed.
    - scenario_column (str): The name of the column in data that identifies the scenario of each episode.
    - output_file (str): The path and name of the CSV file to save the results.

    Returns:
    - Saves the analysis results into a CSV file.
    """
    analysis_results = []
    scenarios = data[scenario_column].unique()

    for scenario in scenarios:
        scenario_data = data[data[scenario_column] == scenario]
        episode_count = len(scenario_data)

        for metric in metric_columns:
            metric_data = scenario_data[metric]

            # Descriptive statistics
            mean = metric_data.mean()
            median = metric_data.median()
            std_dev = metric_data.std()
            min_val = metric_data.min()
            max_val = metric_data.max()

            # Confidence Intervals (95%)
            ci_low, ci_high = stats.norm.interval(0.95, loc=mean, scale=std_dev / np.sqrt(episode_count))

            # Append the results to the analysis_results list
            analysis_results.append({
                scenario_column: scenario,
                'total_episodes': episode_count,
                'metric': metric,
                'mean': mean,
                'median': median,
                'std_dev': std_dev,
                'min': min_val,
                'max': max_val,
                '95%_ci_low': ci_low,
                '95%_ci_high': ci_high
            })

            # Cumulative mean plot (not included in CSV, just visual)
            #cumulative_mean = metric_data.expanding(min_periods=2).mean()
            # plt.figure(figsize=(10, 6))
            # plt.plot(metric_data.index, metric_data, label=f'{metric} per Episode')
            # plt.plot(cumulative_mean.index, cumulative_mean, label=f'Cumulative Mean of {metric}', color='red')
            # plt.title(f'Analysis of {metric} in Scenario {scenario}')
            # plt.xlabel('Episode')
            # plt.ylabel(metric)
            # plt.legend()
            # plt.grid(True)
            # #plt.show()

    # Convert the analysis results list to a DataFrame
    analysis_df = pd.DataFrame(analysis_results)

    # Save the DataFrame to a CSV file
    analysis_df.to_csv(output_file, index=False)
    print(f"Analysis results saved to {output_file}")




eval_mode = 'dc'
base_path = 'apps/models'
experiment_name = 'ma_quadx_chaser_20240204-120343'
model_name = 'model_39500000'

append_csvs(experiment_name=experiment_name, model_name=model_name, base_path=base_path, eval_mode=eval_mode)

input_path = f"{base_path}/{experiment_name}/ep_data/{model_name}/{eval_mode}/{eval_mode}-consolidate_scenarios.csv"
metrics_path = f"{base_path}/{experiment_name}/ep_data/{model_name}/{eval_mode}"


calculate_metrics(input_file_path=input_path, output_file_path=metrics_path)

# Assuming 'data' is your dataframe and it includes a column 'scenario'
# and the metrics you're interested in are in columns like 'metric1', 'metric2', ...
# You would call the function like this:
metric_columns = ['exploded_target', 'downed', 'survived', 'crashes', 'ally_collision', 'exploded_by_ally', 'out_of_bounds', 'timeover', 'max_achieved_speed' ]  # replace with your actual metric column names

data = pd.read_csv(input_path)

# Call the function with your data
descriptive_statistics_path = f"{base_path}/{experiment_name}/ep_data/{model_name}/{eval_mode}/descriptive_statistics.csv"
scenario_metrics_statistics(data=data, metric_columns=metric_columns, scenario_column='scenario', output_file=descriptive_statistics_path)


