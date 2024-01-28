from modules.plotting import plot_tensorboard_data, plot_same_tag_across_experiments

# Do a plot for selected TB tags for each run of one experiment
#plot_tensorboard_data('apps/models/ma_quadx_chaser_20240127-193805/logs', selected_tags=['ep_mean_rew'],  y_axis_range=None)

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
plot_same_tag_across_experiments(logs_dirs=logs_dirs,
                                 tag_to_plot=selected_tag,
                                 experiment_aliases=experiment_aliases,
                                 y_axis_range=None)