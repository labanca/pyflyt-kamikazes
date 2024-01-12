import numpy as np
import yaml
import csv


def save_agg_dict_to_csv(data, file_path):
    with open(file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        # Write the header
        header = ["Experiment", "ep_mean_rew", "ep_max_rew", "ep_min_rew", "ep_std_rew",
                  "ep_mean_len", "ep_max_len", "ep_min_len", "ep_std_len"]
        csv_writer.writerow(header)

        # Write the data
        for experiment, values in data.items():
            row = [experiment,
                   values["ep_mean_rew"], values["ep_max_rew"], values["ep_min_rew"], values["ep_std_rew"],
                   values["ep_mean_len"], values["ep_max_len"], values["ep_min_len"], values["ep_std_len"]]
            csv_writer.writerow(row)


def save_dict_to_csv(data, csv_file):

        fieldnames = data.keys()

        with open(csv_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write the header
            writer.writeheader()

            # Write the data
            writer.writerow(data)




def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    data['env_kwargs']['start_pos'], data['env_kwargs']['start_orn'], data['env_kwargs'][
        'formation_center'] = generate_start_pos_orn(**data['spawn_settings'])

    data['env_kwargs']['flight_dome_size'] = (
            data['spawn_settings']['lw_spawn_radius'] +
            data['spawn_settings']['lm_spawn_radius'] +
            data['spawn_settings']['lw_center_bounds'] +
            data['spawn_settings']['lm_center_bounds']
    )

    return data.get('spawn_settings', {}), data.get('env_kwargs', {}), data.get('train_kwargs', {})


def save_dicts_to_yaml(spawn_settings, env_kwargs, train_kwargs, yaml_file_path):
    data = {
        'spawn_settings': spawn_settings,
        'env_kwargs': env_kwargs,
        'train_kwargs': train_kwargs
    }

    with open(yaml_file_path, 'w') as file:
        yaml.add_representer(np.ndarray, numpy_representer)
        yaml.dump(data, file, default_flow_style=False)


def numpy_representer(dumper, data):
    if isinstance(data, np.ndarray):
        return dumper.represent_list(data.tolist())
    return dumper.represent_scalar('tag:yaml.org,2002:python/none', '')


def generate_start_pos_orn(seed=None, lw_center_bounds=5.0, lw_spawn_radius=1.0, num_lw=3, min_z=1.0,
                           lm_center_bounds=5, lm_spawn_radius=10, num_lm=3,):

    np_random = np.random.RandomState(seed=seed)
    lw_formation_center = [np.random.uniform(-lw_center_bounds, lw_center_bounds),
                           np.random.uniform(-lw_center_bounds, lw_center_bounds),
                           np.random.uniform(min_z, lw_center_bounds + min_z)]

    start_pos_lw = generate_formation_pos(lw_formation_center, num_lw, lw_spawn_radius)
    start_orn_lw = np.zeros_like(start_pos_lw)

    lm_spawn_center = [np.random.uniform(-lm_center_bounds, lm_center_bounds),
                       np.random.uniform(-lm_center_bounds, lm_center_bounds),
                       np.random.uniform(min_z, lm_center_bounds + min_z)]

    start_pos_lm = generate_random_coordinates(lw_formation_center, lw_center_bounds, lw_spawn_radius,
                                                               lm_spawn_center, lm_spawn_radius, num_lm, min_z)

    start_orn_lm = (np_random.rand(num_lm, 3) - 0.5) * 2.0 * np.array([1.0, 1.0, 2 * np.pi])

    return np.concatenate([start_pos_lm, start_pos_lw]), np.concatenate([start_orn_lm, start_orn_lw]), lw_formation_center

def generate_random_coordinates(lw_formation_center, lw_center_bounds, lw_spawn_radius,
                                lm_spawn_center, lm_spawn_radius, num_lm,  min_z):
    # Ensure the formation center and spawn center are NumPy arrays
    lw_formation_center = np.array(lw_formation_center)
    lm_spawn_center = np.array(lm_spawn_center)

    # Generate random coordinates within the specified spawn radius and above the minimum z
    lm_coordinates = []
    while len(lm_coordinates) < num_lm:
        x = np.random.uniform(low=lm_spawn_center[0] - lm_spawn_radius, high=lm_spawn_center[0] + lm_spawn_radius)
        y = np.random.uniform(low=lm_spawn_center[1] - lm_spawn_radius, high=lm_spawn_center[1] + lm_spawn_radius)
        z = np.random.uniform(low=min_z, high=lm_spawn_center[2] + lm_spawn_radius)

        # Check if the generated coordinates are outside the exclusion area of the lw formation
        lm_distance = np.linalg.norm(lw_formation_center[:2] - np.array([x, y]))
        if lm_distance > lw_center_bounds:
            lm_coordinates.append([x, y, z])

    return np.array(lm_coordinates)

def generate_formation_pos( formation_center, num_drones, radius=0.5, min_z = 1.0):
    # Ensure the formation center is a NumPy array
    formation_center = np.array(formation_center)

    # Generate angles evenly distributed around a circle
    angles = np.linspace(0, 2 * np.pi, num_drones, endpoint=False)

    # Calculate drone positions in a radial formation
    x_positions = formation_center[0] + radius * np.cos(angles)
    y_positions = formation_center[1] + radius * np.sin(angles)

    # Set z coordinates to zero (you can modify this based on your specific requirements)
    z_positions = formation_center[2] + np.zeros_like(x_positions)

    # Combine x, y, and z coordinates into a 3D array
    drone_positions = np.column_stack((x_positions, y_positions, z_positions))

    return np.array(drone_positions)