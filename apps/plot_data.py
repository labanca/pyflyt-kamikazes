import pandas as pd

from modules.plotting import plot_tensorboard_data, plot_same_tag_across_experiments, plot_tag_across_runs, \
    create_boxplot, create_cdf_plot, create_histogram, create_boxplot_combined, plot_perfomance_metric_barchart, \
    load_progress_data, plot_progress_data

base_path = "apps/models"

# Load data from multiple experiments
experiment_data = []


experiment_data.append(load_progress_data('apps/models/ma_quadx_chaser_20240204-120343/logs', 'model 1'))
#experiment_data.append(load_progress_data('apps/models/ma_quadx_chaser_20240205-180813/logs', 'train 2'))
experiment_data.append(load_progress_data('apps/models/ma_quadx_chaser_20240210-023412/logs', 'model 2'))
experiment_data.append(load_progress_data('apps/models/ma_quadx_chaser_20240210-232343/logs', 'model 3'))
experiment_data.append(load_progress_data('apps/models/ma_quadx_chaser_20240212-021804/logs', 'model 4'))
experiment_data.append(load_progress_data('apps/models/ma_quadx_chaser_20240210-023412/logs', 'model 5'))



# Combine all experiments data into a single DataFrame
all_experiment_data = pd.concat(experiment_data, ignore_index=True)


# Plot the aggregated data
plot_progress_data(all_experiment_data)











experiment_name = "ma_quadx_chaser_20240213-082419"
model_name = 'ma_quadx_chaser-60135040.zip'

experiment_path = f"{base_path}/{experiment_name}"

# Do a plot for selected TB tags for each run of one experiment
logs_dir = f'{experiment_path}/logs'
run_aliases = {
    'ma_quadx_chaser-10000000': 'Phase 1',
    'ma_quadx_chaser-20027008': 'Phase 2',
    'ma_quadx_chaser-30054016': 'Phase 2',
    'ma_quadx_chaser-40081024': 'Phase 3'
}
selected_tags = ['ep_mean_rew']
#plot_tensorboard_data(logs_dir,selected_tags=selected_tags, run_aliases=run_aliases, y_axis_range=None)


# compare TB tags from different experiments
logs_dirs = [
    'apps/models/ma_quadx_chaser_20240127-193805/logs',
    'apps/models/ma_quadx_chaser_20240127-140521/logs',
    # Add more experiment log directories as needed
]

experiment_aliases = {
    'apps/models/ma_quadx_chaser_20240127-193805/logs': 'Experiment A',
    'apps/models/ma_quadx_chaser_20240127-140521/logs': 'Experiment B',
    # Add more aliases as needed
}
selected_tag = 'ep_mean_rew'
#plot_same_tag_across_experiments(logs_dirs=logs_dirs,
                                 # tag_to_plot=selected_tag,
                                 # experiment_aliases=experiment_aliases,
                                 # y_axis_range=None)

#plot runs from a single experiment
logs_dir = 'apps/models/ma_quadx_chaser_20240127-193805/logs'
selected_tag = 'ep_mean_rew'
# Optional: Define run aliases if you want custom names for runs, otherwise runs will be named "Run 1", "Run 2", etc.
run_aliases = None #{
    #'ma_quadx_chaser-15000000': 'Initial Run',
    #'ma_quadx_chaser-17304000': 'Second Run',
    # Add more run aliases as needed
#}
#plot_tag_across_runs(logs_dir, selected_tag, y_axis_range=None, run_aliases=run_aliases)


"""----------------------------------------------"""
base_path = 'apps\\models'
experiment_name = "ma_quadx_chaser_20240204-120343" #'ma_quadx_chaser_20240128-221717'
model_name = 'model_39500000' #'ma_quadx_chaser-15261776'
eval_modes  = ['rl', 'dc']
column_name = 'elapsed_time'
scenarios = ['5x5','5x5', '10x5', '10x5', '15x5 ', '15x5 ']
#create_boxplot_combined(column_name=column_name, scenarios=scenarios,
#                base_path=base_path, experiment_name=experiment_name, model_name=model_name, eval_modes=eval_modes)

#create_histogram(column_name=column_name, scenarios=scenarios, base_path=base_path, experiment_name=experiment_name, model_name=model_name, eval_mode=eval_mode)

#plot_perfomance_metric_barchart()