import datetime
import os
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from pathlib import Path

matplotlib.use('TkAgg')

from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import seaborn as sns

# Set the appearance settings for all plots
plt.rc('text', usetex=True)
plt.rcParams.update({'font.family': 'serif'})
plt.rcParams.update({'font.size': 16})


def create_boxplot(base_path, experiment_name, model_name, eval_mode, column_name):
    """
    Create a box plot for the specified column from a CSV file located at a constructed path,
    grouped by scenario and episodes.

    :param base_path: Base path to the experiment data
    :param experiment_name: Name of the experiment
    :param model_name: Name of the model
    :param eval_mode: Evaluation mode (part of the file path)
    :param column_name: Name of the column to create box plot for
    """
    # Construct the file path
    file_path = f"{base_path}/{experiment_name}/ep_data/{model_name}/{eval_mode}/consolidate_scenarios.csv"

    # Read the CSV file
    df = pd.read_csv(file_path)

    # Check if the column_name exists in the DataFrame
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame.")

    # Create a boxplot grouped by scenario
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='scenario', y=column_name, data=df)

    # Set plot titles and labels
    plt.title(f'Boxplot of {column_name} grouped by scenario')
    plt.xlabel('Scenario')
    plt.ylabel(column_name)
    plt.xticks(rotation=45, ha='right')  # Rotate x labels for better readability

    # Display the plot
    plt.tight_layout()
    plt.show()


def create_boxplot_combined(base_path, experiment_name, model_name, column_name, scenarios, eval_modes ):
    """
    Create combined box plots for the specified column from CSV files located at constructed paths for 'rl' and 'dc' eval_modes,
    filtered by the given list of scenarios, and displayed side by side.

    :param base_path: Base path to the experiment data
    :param experiment_name: Name of the experiment
    :param model_name: Name of the model
    :param column_name: Name of the column to create box plot for
    :param scenarios: List of scenarios to include in the box plot
    """
    #eval_modes = ['rl', 'dc']
    data_frames = []

    # Read the CSV files and combine them into a single DataFrame
    for eval_mode in eval_modes:
        file_path = f"{base_path}/{experiment_name}/ep_data/{model_name}/{eval_mode}/consolidate_scenarios.csv"
        df = pd.read_csv(file_path)
        df = df[df['scenario'].isin(scenarios)]
        df['eval_mode'] = eval_mode  # Add a column to distinguish between eval_modes
        data_frames.append(df)

    combined_df = pd.concat(data_frames)

    # Check if the column_name exists in the DataFrame
    if column_name not in combined_df.columns:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame.")

    # Create a combined boxplot
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='scenario', y=column_name, hue='eval_mode', data=combined_df, palette="Set2", order=scenarios)

    # Set plot titles and labels
    plt.title(f'Combined Boxplot of {column_name} for RL and DC Eval Modes')
    plt.xlabel('Scenario')
    plt.ylabel(column_name)
    plt.legend(title='Eval Mode')
    plt.xticks(rotation=45, ha='right')  # Rotate x labels for better readability

    # Display the plot
    plt.tight_layout()
    plt.show()


def create_histogram(base_path, experiment_name, model_name, eval_mode, column_name, scenarios, bins=6):
    """
    Create a histogram for the specified column from a CSV file located at a constructed path,
    filtered by the given list of scenarios.

    :param base_path: Base path to the experiment data
    :param experiment_name: Name of the experiment
    :param model_name: Name of the model
    :param eval_mode: Evaluation mode (part of the file path)
    :param column_name: Name of the column to create histogram for
    :param scenarios: List of scenarios to include in the histogram
    :param bins: Number of bins for the histogram (default is 10)
    """
    # Construct the file path
    file_path = f"{base_path}/{experiment_name}/ep_data/{model_name}/{eval_mode}/consolidate_scenarios.csv"

    # Read the CSV file
    df = pd.read_csv(file_path)

    # Filter the DataFrame based on the provided scenarios
    df = df[df['scenario'].isin(scenarios)]

    # Check if the column_name exists in the DataFrame
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame.")

    # Extract the data for the specified column
    data = df[column_name]

    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, edgecolor='black')
    plt.title(f'Histogram of {column_name} for Selected Scenarios')
    plt.xlabel(column_name)
    plt.ylabel('Frequency')

    # Display the plot
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def create_cdf_plot(base_path, experiment_name, model_name, eval_mode, column_name, scenarios):
    """
    Create a Cumulative Distribution Function (CDF) plot for the specified column from a CSV file located at a constructed path,
    filtered by the given list of scenarios.

    :param base_path: Base path to the experiment data
    :param experiment_name: Name of the experiment
    :param model_name: Name of the model
    :param eval_mode: Evaluation mode (part of the file path)
    :param column_name: Name of the column to create CDF plot for
    :param scenarios: List of scenarios to include in the plot
    """
    # Construct the file path
    file_path = f"{base_path}/{experiment_name}/ep_data/{model_name}/{eval_mode}/consolidate_scenarios.csv"

    # Read the CSV file
    df = pd.read_csv(file_path)

    # Filter the DataFrame based on the provided scenarios
    df = df[df['scenario'].isin(scenarios)]

    # Check if the column_name exists in the DataFrame
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame.")

    # Extract the data for the specified column
    data = df[column_name]

    # Calculate the CDF values
    data_sorted = np.sort(data)
    cdf = np.arange(len(data_sorted)) / float(len(data_sorted) - 1)

    # Plotting the CDF
    plt.figure(figsize=(10, 6))
    plt.plot(data_sorted, cdf, marker='.', linestyle='none', )


    plt.title(f'Cumulative Distribution Function (CDF) of {column_name} for {str(scenarios)}')
    plt.xlabel(column_name)
    plt.ylabel('CDF')

    for x_val, y_val in zip(data_sorted, cdf):
        plt.annotate(f"{y_val:.2f}",  # format to 2 decimal places
                     (x_val, y_val),
                     textcoords="offset points",  # how to position the text
                     xytext=(0, 10),  # distance from text to points (x,y)
                     ha='center')  # horizontal alignment can be left, right or center

    # Display the plot
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_tensorboard_data(logs_dir, selected_tags=None, y_axis_range=None, run_aliases=None):
    """
    Plots TensorBoard data for the selected tags with an optional y-axis range, specified appearance settings,
    and aliases for run names.

    :param logs_dir: The directory containing TensorBoard log files.
    :param selected_tags: An optional list of tags to plot. If None, all tags are plotted.
    :param y_axis_range: A tuple (min, max) for the y-axis range. If None, auto range is set.
    :param run_aliases: A dictionary mapping run directory names to their respective aliases.
    """
    plt.figure(figsize=(10, 5))  # Set a larger figure size for clarity

    # Initialize max and min values for y if auto-ranging is required
    global_y_min = float('inf')
    global_y_max = float('-inf')

    experiment_name = Path(logs_dir).parent.name
    logs_path = Path(logs_dir)

    # Iterate over each training run directory
    for run_dir in logs_path.iterdir():
        if run_dir.is_dir():
            run_name = run_aliases.get(run_dir.name, run_dir.name) if run_aliases else run_dir.name

            # List all event files in the run directory
            event_files = list(run_dir.glob('events.out.tfevents.*'))

            # Load each of the event files
            for event_file in event_files:
                # Setting up an EventAccumulator instance to load the log data
                event_acc = EventAccumulator(str(event_file))
                event_acc.Reload()  # Load the file

                # Fetching scalar data (assuming 'Scalars' tab in TensorBoard)
                scalar_tags = event_acc.Tags()['scalars']

                # Filter tags if selected_tags is provided
                if selected_tags is not None:
                    scalar_tags = [tag for tag in scalar_tags if tag in selected_tags]

                # Plot each scalar tag
                for tag in scalar_tags:
                    # Convert to DataFrame
                    df = pd.DataFrame(event_acc.Scalars(tag))
                    df.drop('wall_time', axis=1, inplace=True)  # Drop wall_time column

                    # Plotting
                    plt.plot(df['step'], df['value'], label=f"{run_name} - {tag}", linewidth=2)

                    # Update global y-axis range
                    local_min = df['value'].min()
                    local_max = df['value'].max()
                    global_y_min = min(global_y_min, local_min)
                    global_y_max = max(global_y_max, local_max)

    # Apply the global y-axis range if it wasn't set
    if y_axis_range is None:
        y_axis_range = (global_y_min, global_y_max)

    plt.legend(loc='best', fontsize='small')
    plt.title(f"Episode Mean Reward per Phase")
    plt.xlabel('Step')
    plt.ylabel(selected_tags)
    plt.ylim(y_axis_range)
    plt.grid(True)
    plt.tight_layout()

    fig_filename = Path(logs_path.parent, 'plots','ep_mean_reward-phases.png')
    fig_filename.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_filename, format ='png', dpi=600)

    plt.show()


def plot_same_tag_across_experiments(logs_dirs, tag_to_plot, experiment_aliases, y_axis_range=None):
    """
    Plots the specified tag across different experiments in the same graph, using provided aliases for the experiments
    and automatically generated aliases for the runs.

    :param logs_dirs: A list of directories containing TensorBoard log files for different experiments.
    :param tag_to_plot: The specific tag to plot.
    :param experiment_aliases: A dictionary mapping log directories to their respective experiment aliases.
    :param y_axis_range: An optional tuple (min, max) for the y-axis range. If None, auto range is set.
    """
    plt.figure(figsize=(10, 5))  # Set a larger figure size for clarity

    # Initialize max and min values for y if auto-ranging is required
    global_y_min = float('inf')
    global_y_max = float('-inf')

    for logs_dir in logs_dirs:
        experiment_alias = experiment_aliases.get(logs_dir, 'Unknown Experiment')  # Get experiment alias
        logs_path = Path(logs_dir)

        run_counter = 1  # Initialize a counter for the run names within each experiment

        for run_dir in logs_path.iterdir():
            if run_dir.is_dir():
                run_name = f"Run {run_counter}"  # Generate run name
                run_counter += 1
                event_files = list(run_dir.glob('events.out.tfevents.*'))

                for event_file in event_files:
                    event_acc = EventAccumulator(str(event_file))
                    event_acc.Reload()
                    if tag_to_plot in event_acc.Tags()['scalars']:
                        df = pd.DataFrame(event_acc.Scalars(tag_to_plot))
                        df.drop('wall_time', axis=1, inplace=True)
                        plt.plot(df['step'], df['value'], label=f"{experiment_alias} - {run_name}", linewidth=2)

                        # Update global y-axis range
                        local_min = df['value'].min()
                        local_max = df['value'].max()
                        global_y_min = min(global_y_min, local_min)
                        global_y_max = max(global_y_max, local_max)

    # Apply the global y-axis range if it wasn't set
    if y_axis_range is None:
        y_axis_range = (global_y_min, global_y_max)

    plt.legend(loc='best', fontsize='small')
    plt.title(f"Comparison of '{tag_to_plot}' Across Experiments")
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.ylim(y_axis_range)
    plt.grid(True)
    plt.tight_layout()

    plt.show()


def plot_tag_across_runs(logs_dir, tag_to_plot, y_axis_range=None, run_aliases=None):
    """
    Plots the specified tag from different runs within the same experiment on a single graph.

    :param logs_dir: The directory containing TensorBoard log files for a specific experiment.
    :param tag_to_plot: The specific tag to plot.
    :param y_axis_range: An optional tuple (min, max) for the y-axis range. If None, auto range is set.
    :param run_aliases: An optional dictionary mapping run directories to their respective run aliases.
    """
    plt.figure(figsize=(10, 5))  # Set a larger figure size for clarity

    # Initialize max and min values for y if auto-ranging is required
    global_y_min = float('inf')
    global_y_max = float('-inf')

    experiment_name = Path(logs_dir).parent.name
    logs_path = Path(logs_dir)
    run_counter = 1  # Initialize a counter for the run names within the experiment

    for run_dir in logs_path.iterdir():
        if run_dir.is_dir():
            run_name = f"Run {run_counter}" if not run_aliases else run_aliases.get(run_dir.name, f"Run {run_counter}")
            run_counter += 1
            event_files = list(run_dir.glob('events.out.tfevents.*'))

            for event_file in event_files:
                event_acc = EventAccumulator(str(event_file))
                event_acc.Reload()
                if tag_to_plot in event_acc.Tags()['scalars']:
                    df = pd.DataFrame(event_acc.Scalars(tag_to_plot))
                    df.drop('wall_time', axis=1, inplace=True)
                    plt.plot(df['step'], df['value'], label=f"{run_name}", linewidth=2)

                    # Update global y-axis range
                    local_min = df['value'].min()
                    local_max = df['value'].max()
                    global_y_min = min(global_y_min, local_min)
                    global_y_max = max(global_y_max, local_max)

    # Apply the global y-axis range if it wasn't set
    if y_axis_range is None:
        y_axis_range = (global_y_min, global_y_max)

    plt.legend(loc='best', fontsize='small')
    plt.title(f"{experiment_name} - Comparison of '{tag_to_plot}' Across Runs")
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.ylim(y_axis_range)
    plt.grid(True)
    plt.tight_layout()

    plt.show()




def plot_agent_rewards(filename, agent_id):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(filename)

    # Filter data for the specified agent
    agent_data = df[df['agent_id'] == agent_id]

    # Plotting
    plt.figure(figsize=(12, 8))

    plt.plot(agent_data['elapsed_time'], agent_data['rew_closing_distance'], label='Closing Distance')
    plt.plot(agent_data['elapsed_time'], agent_data['rew_engaging_enemy'], label='Engaging Enemy')
    plt.plot(agent_data['elapsed_time'], agent_data['rew_speed_magnitude'], label='Speed Magnitude')

    plt.xlabel('Elapsed Time')
    plt.ylabel('Rewards')
    plt.title(f'Rewards Over Time for Agent {agent_id}')
    plt.legend()
    plt.grid(True)
    plt.show()

