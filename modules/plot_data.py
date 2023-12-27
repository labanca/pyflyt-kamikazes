import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

def plot_rewards_data(self, filename):


    # Read the CSV file into a DataFrame
    df = pd.read_csv(filename)

    # Plotting
    plt.figure(figsize=(10, 6))

    for agent_id in df['agent_id'].unique():
        agent_data = df[df['agent_id'] == agent_id]
        plt.plot(agent_data['elapsed_time'], agent_data['rew_closing_distance'], label=f'Agent {agent_id}')

    plt.xlabel('Elapsed Time')
    plt.ylabel('Reward - Closing Distance')
    plt.title('Rewards Over Time')
    plt.legend()
    plt.grid(True)
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

# Example usage:
plot_agent_rewards('C:/projects/pyflyt-kamikazes/apps/reward_data.csv', agent_id=0)

