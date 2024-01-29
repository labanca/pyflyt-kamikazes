from modules.plotting import plot_tensorboard_data, plot_same_tag_across_experiments, plot_tag_across_runs

# Do a plot for selected TB tags for each run of one experiment
logs_dir = 'apps/models/ma_quadx_chaser_20240127-193805/logs'
run_aliases = {
    'ma_quadx_chaser-15000000': 'Phase 1',
    'ma_quadx_chaser-17304000': 'Phase 2',
    'ma_quadx_chaser-19080000': 'Phase 3'

}
selected_tags = ['ep_mean_rew']
plot_tensorboard_data(logs_dir,selected_tags=selected_tags, run_aliases=run_aliases, y_axis_range=None)


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