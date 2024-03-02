import csv
from glob import glob
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.use('TkAgg')

def load_progress_data(logs_dir, experiment_alias):
    """
    Recursively search for progress.csv files in the experiment's log directory, concatenate them,
    and return as a pandas DataFrame.
    """
    all_data = []
    logs_path = Path(logs_dir)

    # Search for all progress.csv files within subdirectories of the logs directory
    for progress_file in logs_path.rglob('progress.csv'):
        df = pd.read_csv(progress_file)
        df['Experiment'] = experiment_alias  # Add a column for the experiment alias
        all_data.append(df)

    # Concatenate all found data
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        print(f"No progress.csv files found in {logs_dir}")
        return pd.DataFrame()  # Return an empty DataFrame if no files found


trials_path = glob("*/", recursive=True)


# Assuming trials_path is a list of paths
all_data = []

all_data.append(load_progress_data('apps/models/ma_quadx_chaser_20240204-120343/logs', 'train 1'))
#experiment_data.append(load_progress_data('apps/models/ma_quadx_chaser_20240205-180813/logs', 'train 2'))
all_data.append(load_progress_data('apps/models/ma_quadx_chaser_20240210-023412/logs', 'train 3'))
all_data.append(load_progress_data('apps/models/ma_quadx_chaser_20240210-232343/logs', 'train 4'))
all_data.append(load_progress_data('apps/models/ma_quadx_chaser_20240212-021804/logs', 'train 5'))
all_data.append(load_progress_data('apps/models/ma_quadx_chaser_20240210-023412/logs', 'train 6'))


# Convert the list of DataFrames into a single DataFrame where each column is a trial
# print(all_data)
combined_data = pd.concat(all_data, axis=1)

combined_data = combined_data.drop('Experiment', axis=1)
# Calculate mean and standard deviation across trials at each time point
mean_performance = combined_data.mean(axis=1)
std_performance = combined_data.std(axis=1)

# Plot mean performance and standard deviation as a background
plt.plot(
    mean_performance.index, mean_performance, label="Mean Performance", color="grey"
)
plt.fill_between(
    mean_performance.index,
    mean_performance - std_performance,
    mean_performance + std_performance,
    color="grey",
    alpha=0.2,
)

# Identify specific trials of interest
# For example, we can highlight the best and worst final performances
final_performances = [data.iloc[-1] for data in all_data]
#best_trial_index = np.argmax(final_performances)
#worst_trial_index = np.argmin(final_performances)

#print(f"Best trial: {best_trial_index}")
#print(f"Worst trial: {worst_trial_index}")
# Highlight the best and worst trials
# plt.plot(
#    all_data[best_trial_index].index,
#    all_data[best_trial_index],
#    label=f"Best: Trial_{best_trial_index}",
#    color="green",
# )

# for data in all_data:
#    plt.plot(data.index, data, alpha=0.7)

# plt.plot(
#     all_data[4].index,
#     all_data[4],
#     label=f"Bayesian Optimizer Trial {6}",
#     color="blue",
# )


# plt.plot(
#    all_data[worst_trial_index].index,
#    all_data[worst_trial_index],
#    label=f"Worst: Trial_{worst_trial_index}",
#    color="red",
# )

# plt.plot(
#    all_data[4].index,
#    all_data[4],
#    label=f"Best in Evaluation: Trial_{4}",
#    color="yellow",
# )

# Optionally, annotate other trials of interest
# for i, data in enumerate(all_data):
#    if i == 4:
#        plt.plot(data.index, data, label=f"Trial {i}", alpha=0.7)

plt.xlabel("TimeSteps")
plt.ylabel("Moving Mean Reward")
# plt.title("Performance Across Trials")
plt.legend()


# Save the figure with a higher DPI
# plt.savefig(
#    "performance_plot.pdf", dpi=300
# )  # Saves the figure in PDF format with 300 DPI

# If you want to save in SVG or EPS format, simply change the file extension
plt.savefig(
    "exp 02 - performance_plot - single 10 times.svg", dpi=300
)  # Saves the figure in SVG format
# plt.savefig('performance_plot.eps', dpi=300)  # Saves the figure in EPS format


plt.show()
