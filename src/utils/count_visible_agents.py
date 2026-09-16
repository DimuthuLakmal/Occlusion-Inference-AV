import os
import pickle
from pathlib import Path
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

filename = "data.pkl"
input_path = "../../data"
annotations_path = "../../data/annotations"

# Check data file exists
index_file_path = Path(os.path.join(input_path, filename))

data_dict = pickle.load(open(index_file_path, "rb"))
annotation_files = [f for f in os.listdir(annotations_path) if f.endswith('.json')]

total_agents = 0
total_moving_agents = 0
num_moving_agent_dict = {}
moving_agents_counts = []
for file in annotation_files:
    key = file.split('.')[0]
    record = data_dict[key]
    num_agents = len(list(record['historical_adjacent_obs'].keys()))  # Shape: (num_agents, history_length, feature_dim)
    total_agents += num_agents

    # Find moving agents
    historical_obs = np.array((list(record['historical_adjacent_obs'].values())))
    speeds_x = historical_obs[:, :, 3]  # Assuming speed in x is at index 3
    speeds_y = historical_obs[:, :, 4]  # Assuming speed in y is at index 4
    # if any agent has non-zero speed at any time step, consider it moving
    moving_agents = np.where(np.any((speeds_x != 0) | (speeds_y != 0), axis=1))[0]
    num_moving_agents = len(moving_agents)
    total_moving_agents += num_moving_agents

    moving_agents_counts.append(num_moving_agents)

    # Fill the num_agent_dict for further analysis if needed
    if num_moving_agents not in num_moving_agent_dict:
        num_moving_agent_dict[num_moving_agents] = 1
    else:
        num_moving_agent_dict[num_moving_agents] += 1

print(f'Total number of agents in the dataset: {total_agents}')
print('average number of agents per frame:', total_agents / len(annotation_files))
print('average number of moving agents per frame:', total_moving_agents / len(annotation_files))

# STD of moving agents per frame
std_moving_agents = np.std(moving_agents_counts)
print('std of number of moving agents per frame:', std_moving_agents)

# Extract keys and values for plotting
# Use list() for compatibility across Python versions
num_moving_agent_keys = list(num_moving_agent_dict.keys())
num_moving_agent_vals = list(num_moving_agent_dict.values())

# Create the bar chart
plt.figure(figsize=(8, 6)) # Optional: adjust figure size
plt.bar(num_moving_agent_keys, num_moving_agent_vals, color='skyblue')
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

# Add labels and title
plt.xlabel('Num of Agents per Scene')
plt.ylabel('Frequency')

# Display the plot
plt.show()
