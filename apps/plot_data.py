from modules.plotting import plot_tensorboard_data, plot_same_tag_across_experiments, plot_tag_across_runs, \
    create_boxplot, create_cdf_plot, create_histogram, create_boxplot_combined

# Do a plot for selected TB tags for each run of one experiment
logs_dir = 'apps/models/ma_quadx_chaser_20240202-014543/logs'
run_aliases = {
    'ma_quadx_chaser-10000000': 'Phase 1',
    'ma_quadx_chaser-20000000': 'Phase 2',
    'ma_quadx_chaser-25000000': 'Phase 3',
    'ma_quadx_chaser-30000000': 'Phase 4'
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


"""----------------------------------------------"""
base_path = 'apps\\models\\bkp 2024-02-03'
experiment_name = "ma_quadx_chaser_20240131-161251" #'ma_quadx_chaser_20240128-221717'
model_name = 'ma_quadx_chaser-24000000' #'ma_quadx_chaser-15261776'
eval_modes  = ['rl', 'dc']
column_name = 'elapsed_time'
scenarios = ['rl_sc3_5x5','dc_sc3_5x5', 'rl_sc8_10x5', 'dc_sc8_10x5', 'rl_sc9_15x5 ', 'dc_sc9_15x5 ']
create_boxplot_combined(column_name=column_name, scenarios=scenarios,
                base_path=base_path, experiment_name=experiment_name, model_name=model_name, eval_modes=eval_modes)

#create_histogram(column_name=column_name, scenarios=scenarios, base_path=base_path, experiment_name=experiment_name, model_name=model_name, eval_mode=eval_mode)