import datetime
import os
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from pathlib import Path

matplotlib.use('TkAgg')

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Set the appearance settings for all plots
plt.rc('text', usetex=True)
plt.rcParams.update({'font.family': 'serif'})
plt.rcParams.update({'font.size': 16})


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
    plt.ylabel('Value')
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

