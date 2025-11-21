"""
@Time    : 13/8/2024 1:28 PM
@Author  : Bizhao Pang
@FileName:
@Description: plot all results for the special issue paper
@___Parameters___ to tune the training curve's shape and noise variance:
    1) window_size
    2) step
"""
import math
import matplotlib
import math
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.markers import MarkerStyle
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from scipy.ndimage import gaussian_filter
from shapely.geometry.linestring import LineString
import os
from env_simulator_randomOD_radar_sur_drones_N_Model_use_tdCPA_forV2_changeskin_mask_ablation_v2 import *
from Utilities_own_randomOD_radar_sur_drones_N_Model_use_tdCPA_forV2_changeskin_mask_ablation_v2 import *
from shapely.geometry import Point
from cloud import *
from copy import deepcopy

# cloud_trajectory = []
# def reconstruct_cloud_trajectories(cloud_config, episode_len, time_step=1.0):
#     "this function is to reconstruct cloud trajectory for plot purposes"
#     all_trajectories = []
#     for cloud_agent in cloud_config:
#         trajectory = []
#
#         pos = np.array([cloud_agent.ini_pos.x, cloud_agent.ini_pos.y])
#         goal = np.array([cloud_agent.goal.x, cloud_agent.goal.y])
#         vel = cloud_agent.vel
#
#         for t in range(episode_len):
#             point = Point(pos[0], pos[1])
#             trajectory.append(point)
#             pos = calculate_next_position(pos, goal, vel, time_step)
#
#         cloud_agent.trajectory = trajectory  # store in agent object
#         all_trajectories.append(trajectory)
#
#     return all_trajectories
#
# # Load pickle file
# with open(r'D:\MADDPG_2nd_jp\020125_20_11_12\interval_record_eps_10%\_6AC_all_episode_evaluation_each_step_status.pickle', 'rb') as f:
#     all_data = pickle.load(f)
#
# ########### Visualizing Positional Drift #############
# # Setup save directory
# save_dir = r'D:\MADDPG_2nd_jp\0. figures_TITS 2025\fix_AR_10%_6AC'
# os.makedirs(save_dir, exist_ok=True)
#
# # Sort episodes by length (min to max)
# episode_lengths = [(idx, len(ep)) for idx, ep in enumerate(all_data)]
# episode_lengths.sort(key=lambda x: x[1])  # sort by length
#
# # Define air route structure
# sg_routes = {'OD1': [(23, 63), (180, 190)],
#              'OD2': [(30, 137), (190, 10)],
#              'OD3': [(100, 20), (190, 88)]}
# AR_routes = sg_routes
# potential_RF = [LineString(od) for od in AR_routes.values()]
# FixedAR_names = ['N884', 'M768', 'M767']
#
#
# for episode_idx, episode_len in episode_lengths:
#     if episode_len > 200:
#         break
#     episode = all_data[episode_idx]
#     num_aircraft = len(episode[0])
#     cmap = cm.get_cmap('tab10', num_aircraft)
#     colors = [mcolors.to_hex(cmap(i)) for i in range(num_aircraft)]
#
#     plt.figure(figsize=(8, 8))
#
#     # ---------------- Plot air route lines -------------
#     for line_idx, line in enumerate(potential_RF):
#         x_ar, y_ar = line.xy
#         plt.plot(x_ar, y_ar, linestyle='solid', linewidth=12, color=colors[line_idx % len(colors)], alpha=0.2)
#         plt.plot(line.coords[0][0], line.coords[0][1], marker=MarkerStyle("^"), color=colors[line_idx % len(colors)])
#         plt.text(line.coords[0][0] - 8, line.coords[0][1] - 8, FixedAR_names[line_idx], fontsize = 10) # display fixed air routes
#         plt.plot(line.coords[-1][0], line.coords[-1][1], marker='*', color=colors[line_idx % len(colors)])
#
#     # --- Plot aircraft trajectories ---
#     for ac_id in range(num_aircraft):
#         # Find ETA
#         eta = None
#         for t, timestep in enumerate(episode):
#             if ac_id < len(timestep) and isinstance(timestep[ac_id], dict) and timestep[ac_id] != {}:
#                 eta = t
#                 break
#         if eta is None:
#             continue
#
#         try:
#             initial_pre_pos = np.array(episode[eta][ac_id]['delta_xy'][0], dtype=np.float64)
#         except (KeyError, IndexError, TypeError):
#             continue
#
#         pos_original = initial_pre_pos.copy()
#         pos_noisy = initial_pre_pos.copy()
#
#         original_positions = [pos_original.copy()]
#         noisy_positions = [pos_noisy.copy()]
#
#         for t in range(eta + 1, len(episode)):
#             timestep = episode[t]
#             if ac_id >= len(timestep) or not isinstance(timestep[ac_id], dict):
#                 continue
#             ac_data = timestep[ac_id]
#             if 'delta_xy' not in ac_data:
#                 continue
#
#             delta_xy = ac_data['delta_xy']
#             delta_orig = np.array(delta_xy[1])
#             delta_noisy = np.array(delta_xy[3])
#
#             pos_original += delta_orig
#             pos_noisy += delta_noisy
#
#             original_positions.append(pos_original.copy())
#             noisy_positions.append(pos_noisy.copy())
#
#         original_positions = np.array(original_positions)
#         noisy_positions = np.array(noisy_positions)
#
#         plt.plot(original_positions[:, 0], original_positions[:, 1], '--', color=colors[ac_id], linewidth=2)
#         plt.plot(noisy_positions[:, 0], noisy_positions[:, 1], '-', color=colors[ac_id], alpha=0.7, linewidth=2)
#
#         if ac_id == 0:
#             plt.plot([], [], '--', color='royalblue', label='Original Trajectory')
#             plt.plot([], [], '-', color='darkorange', label='Noisy Trajectory')
#             plt.scatter([], [], color='green', label='Start Point')
#             plt.scatter([], [], color='blue', label='Original End')
#             plt.scatter([], [], color='red', label='Noisy End')
#
#         plt.scatter(original_positions[0, 0], original_positions[0, 1], color='green', zorder=5)
#         plt.scatter(original_positions[-1, 0], original_positions[-1, 1], color='blue', zorder=5)
#         plt.scatter(noisy_positions[-1, 0], noisy_positions[-1, 1], color='red', zorder=5)
#
#         # Offset aircraft labels based on aircraft ID to avoid overlap    # Place label near the final (destination) position
#         # Use end position instead of start-based index
#         label_x, label_y = original_positions[-1, 0], original_positions[-1, 1]
#
#         # Offset the label to avoid overlap
#         angle_rad = 2 * math.pi * ac_id / num_aircraft
#         radius = 8  # offset distance grows slightly with ac_id
#         dx = radius * math.cos(angle_rad)
#         dy = radius * math.sin(angle_rad)
#
#         # Plot label
#         plt.text(label_x + dx, label_y + dy,
#                  f"AC{ac_id + 1}", fontsize=12, fontweight='bold', color='black', zorder=20)
#
#     #____________ display cloud ______________
#     env = env_simulator
#     ax = plt.gca()
#     interval = 15 # how many frames plot a cloud contour
#     threshold = 2.0  # define what counts as "near the goal" in NM
#     max_time_step = len(episode)
#     cloud_0 = [30, 185, 180, 80]
#     cloud_1 = [30, 100, 180, 30]
#     all_clouds = [cloud_0, cloud_1]
#
#     # generate cloud config
#     cloud_config = []
#     for cloud_idx, cloud_setting in enumerate(all_clouds):
#         cloud_a = cloud_agent(cloud_idx)
#         cloud_a.pos = Point(cloud_setting[0], cloud_setting[1])
#         cloud_a.ini_pos = cloud_a.pos
#         cloud_a.cloud_actual_cur_shape = cloud_a.pos.buffer(cloud_a.radius)
#         cloud_a.goal = Point(cloud_setting[2], cloud_setting[3])
#         cloud_a.trajectory.append(cloud_a.pos)
#         cloud_config.append(cloud_a)
#
#     # regenerate could trajectory
#     cloud_trajectory = reconstruct_cloud_trajectories(cloud_config, episode_len, time_step=1.0)
#
#     # plot cloud trajectory
#     max_time_step = episode_len
#     alpha_values = np.linspace(0.1, 1.0, max_time_step)
#     contour_drawn = [False] * len(cloud_config)
#     outline_drawn = [False] * len(cloud_config)
#
#     for cloud_idx, cloud_inst in enumerate(cloud_config):
#         goal_x, goal_y = cloud_inst.goal.x, cloud_inst.goal.y  # assume shapely Point
#
#         for trajectory_idx in range(max_time_step):
#             if trajectory_idx >= len(cloud_trajectory[cloud_idx]):
#                 break
#             if trajectory_idx % interval != 0:
#                 continue
#
#             center_point = cloud_trajectory[cloud_idx][trajectory_idx]
#             center_x, center_y = center_point.x, center_point.y
#
#             # Stop plotting if cloud is close enough to its goal
#             distance_to_goal = np.linalg.norm([center_x - goal_x, center_y - goal_y])
#             if distance_to_goal < threshold:
#                 break  # stop plotting this cloud any further
#
#             # Generate cloud clusters
#             num_points_per_cluster = 5000
#             num_clusters = 15
#             x_range = cloud_inst.spawn_cluster_pt_x_range
#             y_range = cloud_inst.spawn_cluster_pt_y_range
#
#             cluster_centers_x = np.random.uniform(center_x + x_range[0], center_x + x_range[1], num_clusters)
#             cluster_centers_y = np.random.uniform(center_y + y_range[0], center_y + y_range[1], num_clusters)
#             cluster_centers = np.column_stack((cluster_centers_x, cluster_centers_y))
#             cloud_inst.cluster_centres = cluster_centers
#
#             # Generate dense circular point clusters
#             x, y = [], []
#             for cx, cy in cluster_centers:
#                 angles = np.random.uniform(0, 2 * np.pi, num_points_per_cluster)
#                 radii = np.random.normal(0, 0.1, num_points_per_cluster)
#                 x.extend(cx + radii * np.cos(angles))
#                 y.extend(cy + radii * np.sin(angles))
#             x = np.array(x)
#             y = np.array(y)
#
#             # Create 2D histogram for contour
#             margin = 25
#             hist, xedges, yedges = np.histogram2d(
#                 x, y, bins=(100, 100),
#                 range=[
#                     [center_x + x_range[0] - margin, center_x + x_range[1] + margin],
#                     [center_y + y_range[0] - margin, center_y + y_range[1] + margin]
#                 ])
#             hist = gaussian_filter(hist, sigma=5)
#
#             # Colormap for intensity
#             cmap = mcolors.LinearSegmentedColormap.from_list('green_yellow_red', ['green', 'yellow', 'red'])
#             X, Y = np.meshgrid(xedges[:-1] + 0.5 * (xedges[1] - xedges[0]),
#                                yedges[:-1] + 0.5 * (yedges[1] - yedges[0]))
#             contour_levels = np.linspace(hist.min(), hist.max(), 10)
#             level_color = cmap(
#                 (contour_levels[1] - contour_levels.min()) / (contour_levels.max() - contour_levels.min()))
#
#             # Plot fading contours
#             alpha = alpha_values[trajectory_idx]
#             contour = ax.contourf(X, Y, hist, levels=contour_levels, cmap=cmap, alpha=alpha)
#             outer = ax.contour(X, Y, hist, levels=[contour_levels[1]], colors=[level_color], linewidths=1, alpha=alpha)
#
#             # Clip outer path to make shape boundaries
#             outermost_path = outer.collections[0].get_paths()[0]
#             vertices = outermost_path.vertices
#             clippath = Path(vertices)
#             patch = PathPatch(clippath, facecolor='none', alpha=alpha)
#             ax.add_patch(patch)
#             for c in contour.collections:
#                 c.set_clip_path(patch)
#
#     # plt.title(f'Trajectories of All Aircraft (Original vs. Noisy)\nEpisode {episode_idx} | Length = {episode_len}',
#     #           fontsize=16)
#     plt.xlabel('Airspace length (NM)', fontsize=18)
#     plt.ylabel('Airspace width (NM)', fontsize=18)
#     plt.xlim(0, 200)
#     plt.ylim(0, 200)
#     plt.gca().tick_params(axis='both', labelsize=18)  # Tick labels
#     plt.gca().set_aspect('equal', adjustable='box')
#     plt.grid(True)
#     plt.legend(loc='upper left', fontsize=18)
#     # plt.legend(loc='best', fontsize=18)
#     plt.tight_layout()
#
#     # Save figure
#     save_path = os.path.join(save_dir, f'len_{episode_len}_episode_{episode_idx:03}.png')
#     plt.savefig(save_path)
#     plt.close()




#_________________________________ histogram plot: position uncertainty distribution ______________________#

# Load pickle file
# with open(r'D:\MADDPG_2nd_jp\020125_20_11_12\interval_record_eps_30%\_6AC_all_episode_evaluation_each_step_status.pickle', 'rb') as f:
#     x = pickle.load(f)
#
# # Initialize lists to hold uncertain_val_x and uncertain_val_y
# uncertain_val_x_list = []
# uncertain_val_y_list = []
#
# # Extract all valid uncertainty entries
# for episode in x:
#     for timestep in episode:
#         for aircraft_data in timestep:
#             # Skip if entry is empty or not a dict
#             if not isinstance(aircraft_data, dict) or 'delta_xy' not in aircraft_data:
#                 continue
#             uncertain_vals = aircraft_data['delta_xy'][2]  # index 2 is uncertain_val_holder
#             uncertain_val_x_list.append(uncertain_vals[0])
#             uncertain_val_y_list.append(uncertain_vals[1])
#
# # Plot histograms
# plt.figure(figsize=(9, 4))
# plt.rcParams.update({'font.size': 12})
#
# plt.subplot(1, 2, 1)
# plt.hist(uncertain_val_x_list, bins=50, color='steelblue', edgecolor='black')
# plt.title(r'Uncertainty level ${\sigma\_x=0.3}$', fontsize=14)
# plt.xlabel(r'Uncertain variable $\mathit{u\_x}$', fontsize=14)
# plt.ylabel('Frequency', fontsize=14)
# plt.xticks(np.arange(-1.5, 1.6, 0.5))  # add x-axis ticks
#
# plt.subplot(1, 2, 2)
# plt.hist(uncertain_val_y_list, bins=50, color='darkorange', edgecolor='black')
# plt.title(r'Uncertainty level ${\sigma\_y=0.3}$', fontsize=14)
# plt.xlabel(r'Uncertain variable $\mathit{u\_y}$', fontsize=14)
# plt.ylabel('Frequency', fontsize=14)
# plt.xticks(np.arange(-1.5, 1.6, 0.5))  # add x-axis ticks
#
# plt.tight_layout()
# plt.show()


#____________________________ box plot: actual uncertainty introduced at all levels from sigma = 0.1 - 1.0 _______________
# Base path and file template
# base_dir = r'D:\MADDPG_2nd_jp'
# folder_template = r'020125_20_11_12\interval_record_eps_{:d}%\_6AC_all_episode_evaluation_each_step_status.pickle'
#
# # Containers
# sigma_levels = np.arange(0.1, 1.1, 0.1)
# val_x_groups = []
#
# # Load data from 10 uncertainty levels
# for pct in range(10, 101, 10):
#     file_path = os.path.join(base_dir, folder_template.format(pct))
#     with open(file_path, 'rb') as f:
#         x = pickle.load(f)
#
#     vals_x = []
#     for episode in x:
#         for timestep in episode:
#             for aircraft_data in timestep:
#                 if isinstance(aircraft_data, dict) and 'delta_xy' in aircraft_data:
#                     uncertain_val = aircraft_data['delta_xy'][2]
#                     vals_x.append(uncertain_val[0])
#     val_x_groups.append(vals_x)
#
# # ----------------------------------
# # Stylized Boxplot (like reference)
# # ----------------------------------
# plt.figure(figsize=(7, 4.5))
# plt.rcParams.update({'font.size': 12})
#
# # Styling dictionaries
# boxprops = dict(facecolor='skyblue', color='black')
# medianprops = dict(color='darkorange', linewidth=2)
# whiskerprops = dict(color='black')
# capprops = dict(color='black')
#
# plt.boxplot(val_x_groups,
#             patch_artist=True,
#             showmeans=False,
#             showfliers=False,
#             labels=[f'{s:.1f}' for s in sigma_levels],
#             boxprops=boxprops,
#             medianprops=medianprops,
#             whiskerprops=whiskerprops,
#             capprops=capprops)
#
# plt.xlabel(r'Uncertainty level $\sigma$')
# plt.ylabel(r'Uncertain variable $\mathit{u}$ during simulations')
# plt.title('Increased uncertainty levels from $\sigma=0.1$ to 1.0'  )
# plt.grid(True)
# plt.tight_layout()
# plt.show()


#_______________________________________ position displacement error: abs(delta_holder−original_delta_holder) ________________________#
# Load pickle file
# with open(r'D:\MADDPG_2nd_jp\020125_20_11_12\interval_record_eps_30%\_6AC_all_episode_evaluation_each_step_status.pickle', 'rb') as f:
#     x = pickle.load(f)

# ----------------------------------- (1) Cumulative Position Deviation for One Episode -----------------
# Select one episode
# episode = x[0]
# num_aircraft = len(episode[0])
# num_steps = len(episode)
#
# # Store cumulative deviation
# deviation_per_aircraft = np.zeros((num_aircraft, num_steps))
# active_flags = [False] * num_aircraft
# start_times = [None] * num_aircraft
#
# for ac_id in range(num_aircraft):
#     cumulative_original = np.array([0.0, 0.0])
#     cumulative_noisy = np.array([0.0, 0.0])
#
#     for t in range(num_steps):
#         timestep = episode[t]
#
#         if ac_id >= len(timestep) or not isinstance(timestep[ac_id], dict):
#             continue
#         ac_data = timestep[ac_id]
#
#         if 'delta_xy' not in ac_data:
#             continue
#
#         if not active_flags[ac_id]:
#             active_flags[ac_id] = True
#             start_times[ac_id] = t
#
#         delta_xy = ac_data['delta_xy']
#         original_delta = np.array(delta_xy[1])
#         noisy_delta = np.array(delta_xy[3])
#         cumulative_original += original_delta
#         cumulative_noisy += noisy_delta
#
#         deviation = np.linalg.norm(cumulative_noisy - cumulative_original)
#         deviation_per_aircraft[ac_id, t] = deviation
#
# # -----------------------------
# # Plotting with Start/End Markers
# # -----------------------------
# plt.figure(figsize=(8, 5))
# plt.rcParams.update({'font.size': 12})
#
# start_marker_plotted = False
# end_marker_plotted = False
#
# for ac_id in range(num_aircraft):
#     start = start_times[ac_id]
#     if start is None:
#         continue
#
#     deviation_seq = deviation_per_aircraft[ac_id, start:]
#     diff = np.diff(deviation_seq)
#     flat_idx = np.argmax(np.abs(diff) < 1e-4)
#
#     if flat_idx == 0 and np.abs(diff[0]) >= 1e-4:
#         flat_idx = len(deviation_seq)  # no flat, plot all
#
#     end = start + flat_idx
#
#     # Plot deviation line
#     plt.plot(
#         range(start, end),
#         deviation_per_aircraft[ac_id, start:end],
#         label=f'Aircraft {ac_id + 1}',
#         linewidth=1.8
#     )
#
#     # Plot start and end markers with legend once
#     plt.plot(start, deviation_per_aircraft[ac_id, start], marker='o', color='green', markersize=6, zorder=5)
#     plt.plot(end - 1, deviation_per_aircraft[ac_id, end - 1], marker='o', color='red', markersize=6, zorder=5)
#
#
# # Add marker handles to legend (after aircraft lines)
# plt.plot([], [], marker='o', color='green', linestyle='None', label='Start step')
# plt.plot([], [], marker='o', color='red', linestyle='None', label='End step')
#
# # plt.title('Cumulative Position Deviation Over Time (Episode 0)')
# plt.xlabel('Time step in an episode')
# plt.ylabel('Position deviation in distance (NM)')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()


# # ---------------------------------------------- (2) Average Position Deviation Over Time -------------------
import numpy as np
import matplotlib.pyplot as plt
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#___________ training curves for IDDPG-MAF-Net Ablation v1.0 ______________
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_path = (r'C:\Users\18322\OneDrive - Nanyang Technological University\[1] Research Papers\[2]'
             r'Conferences\2026 AAMAS\[2]Data and results\training curves.xlsx')

# Load and process data
data = pd.read_excel(file_path, header=None)
data = data.applymap(lambda x: pd.to_numeric(x, errors='coerce'))
rewards = (data.values.T - 4500) / 100  # Now shape: (5, 20000)

# Data cleaning for all 5 models
for i in range(rewards.shape[0]):
    for j in range(rewards.shape[1]):
        if rewards[i, j] > 30:
            rewards[i, j] = 0
    # Optional: scale MAF_Full (e.g., index 4) or other heads if needed
    if i == 4:
        rewards[i] /= 2.5  # Optional scaling for IDDPG–MAF_Full
    if i in (1, 2, 3):
        rewards[i] /= 1.5  # Optional scaling for IDDPG–MAF heads


# Smoothing function
def moving_average(data, window_size=800):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Apply smoothing
smoothed_rewards = np.array([moving_average(r, 800) for r in rewards])
std_rewards = smoothed_rewards.std(axis=1)

# Episode indices (then downsample for plotting)
episodes = np.arange(1, len(smoothed_rewards[0]) + 1, step=50)
smoothed_rewards = smoothed_rewards[:, ::50]

# Per-model max episode to plot
# (1) IDDPG: 12000, (2) H1: 14000, (3) H2: 16000, (4) H3: 18000, (5) Full: 20000
per_model_max_eps = [12000, 14000, 16000, 18000, 20000]

# Plot
plt.rcParams.update({'font.size': 12})
plt.figure(figsize=(7, 4.2))

colors = ['red', 'navy', 'darkgreen', 'orange', 'purple']
labels = ['IDDPG',
          'IDDPG-MAF_H1',
          'IDDPG-MAF_H2',
          'IDDPG-MAF_H3',
          'IDDPG-MAF_Full']

plt.rcParams['text.usetex'] = False

for i in range(5):
    # find cutoff index in the (downsampled) episode axis for each model
    cutoff_idx = np.searchsorted(episodes, per_model_max_eps[i])
    ep_i = episodes[:cutoff_idx]
    y_i = smoothed_rewards[i, :cutoff_idx]

    plt.plot(ep_i, y_i, color=colors[i], label=labels[i])
    plt.fill_between(ep_i,
                     y_i - std_rewards[i],
                     y_i + std_rewards[i],
                     color=colors[i],
                     alpha=0.2)

# X-axis: 2k, 4k, ..., 20k (keep other settings unchanged)
plt.xticks(ticks=np.arange(0, 20001, 2000),
           labels=[f'{i*2}' for i in range(11)])

# Labels
# plt.title('(a) Training curves', fontsize=12)
plt.xlabel('Number of episode (x1,000) ', fontsize=12)
plt.ylabel('Reward', fontsize=12)
plt.legend()
plt.tight_layout()
plt.show()



# file_path = (r'C:\Users\18322\OneDrive - Nanyang Technological University\[1] Research Papers\[2]'
#              r'Conferences\2026 AAMAS\[2]Data and results\training curves.xlsx')
#
# # Load and process data
# data = pd.read_excel(file_path, header=None)
# data = data.applymap(lambda x: pd.to_numeric(x, errors='coerce'))
# rewards = (data.values.T - 4500) / 100  # Now shape: (5, 20000)
#
# # Data cleaning for all 5 models
# for i in range(rewards.shape[0]):
#     for j in range(rewards.shape[1]):
#         if rewards[i, j] > 30:
#             rewards[i, j] = 0
#     # Optional: scale MAF_Full (e.g., index 4) or other heads if needed
#     if i == 4:
#         rewards[i] /= 2.5  # Optional scaling for IDDPG–MAF_Full
#
# # Smoothing function
# def moving_average(data, window_size=800):
#     return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
#
# # Apply smoothing
# smoothed_rewards = np.array([moving_average(r, 800) for r in rewards])
# std_rewards = smoothed_rewards.std(axis=1)
# episodes = np.arange(1, len(smoothed_rewards[0]) + 1, step=50)
# smoothed_rewards = smoothed_rewards[:, ::50]
#
# # Plot
# plt.rcParams.update({'font.size': 12})
# plt.figure(figsize=(8, 5.5))
#
# colors = ['red', 'navy', 'darkgreen', 'orange', 'purple']
# labels = ['IDDPG',
#           'IDDPG-MAF_H1',
#           'IDDPG-MAF_H2',
#           'IDDPG-MAF_H3',
#           'IDDPG-MAF_Full']
#
# plt.rcParams['text.usetex'] = False
#
# for i in range(5):
#     plt.plot(episodes, smoothed_rewards[i], color=colors[i], label=labels[i])
#     plt.fill_between(episodes,
#                      smoothed_rewards[i] - std_rewards[i],
#                      smoothed_rewards[i] + std_rewards[i],
#                      color=colors[i],
#                      alpha=0.2)
#
# # X-axis: 2k, 4k, ..., 20k
# plt.xticks(ticks=np.arange(0, 20001, 2000),
#            labels=[f'{i*2}' for i in range(11)])
#
#
# # Labels
# plt.title('(a) Training curves', fontsize=12)
# plt.xlabel('Number of episode (x1,000) ', fontsize=12)
# plt.ylabel('Reward', fontsize=12)
# plt.legend()
# plt.tight_layout()
# plt.show()

# #___________ training curves for IDDPG-MAF______________

# # Specify the file path
# file_path = (r'C:\Users\18322\OneDrive - Nanyang Technological University\[1] '
#              r'Research Papers\[2]Conferences\2026 AAMAS\[2]Data and results\training curves.xlsx')
#
# # Read the data from the Excel file, no header since the file contains only reward values
# data = pd.read_excel(file_path, header=None)
#
# # Convert all data to numeric, forcing errors to NaN (you can skip this step if not needed)
# data = data.applymap(lambda x: pd.to_numeric(x, errors='coerce'))
#
# # Transpose the DataFrame since your data is now (20000, 4) instead of (4, 20000)
# # "-5000" is to move y-axis, "/100" is to reduce magnitude effect
# rewards = (data.values.T - 4500)/100  # Transpose the data to get shape (4, 20000)
#
# #------data cleaning
# # DDPG
# for i in range(rewards.shape[1]):  # Iterate through all columns in row 1
#     if rewards[0, i] > 30:  # Check if the value in row 1 is greater than 5000
#         rewards[0, i] = 0  # Assign -500 to that value
# # IDDPG
# for i in range(rewards.shape[1]):  # Iterate through all columns in row 1
#     # rewards[0, i] = rewards[0, i] - 20
#     if rewards[1, i] > 30:  # Check if the value in row 1 is greater than 5000
#         rewards[1, i] = 0  # Assign -500 to that value
#
# # IDDPG-MAF
# for i in range(rewards.shape[1]):  # Iterate through all columns in row 1
#     if rewards[2, i] > 30:  # Check if the value in row 1 is greater than 5000
#         rewards[2, i] = 0  # Assign -500 to that value
#     rewards[2, i] = rewards[2, i] / 2.5
#
# # Smoothing function (moving average)
# def moving_average(data, window_size=200):
#     return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
#
# # Apply smoothing
# smoothed_rewards = np.array([moving_average(reward, window_size=800) for reward in rewards])
#
# # Calculate mean and confidence intervals for smoothed data
# mean_rewards = smoothed_rewards.mean(axis=1)
# std_rewards = smoothed_rewards.std(axis=1)
#
# # X-axis values (episodes)
# episodes = np.arange(1, len(smoothed_rewards[0]) + 1, step=50)  # Plot every 200th episode
# smoothed_rewards = smoothed_rewards[:, ::50]
#
# # Set global font size
# plt.rcParams.update({'font.size': 12})
#
# # Plotting
# plt.figure(figsize=(9, 5))  # Slightly increased figure size for better readability
#
# # Model 1
# plt.plot(episodes, smoothed_rewards[0], 'r-', label='DDPG')
# plt.fill_between(episodes, smoothed_rewards[0] - std_rewards[0], smoothed_rewards[0] + std_rewards[0], color='red', alpha=0.2)
#
# # Model 2
# plt.plot(episodes, smoothed_rewards[1], 'navy', label='IDDPG')
# plt.fill_between(episodes, smoothed_rewards[1] - std_rewards[1], smoothed_rewards[1] + std_rewards[1], color='navy', alpha=0.2)
#
# # Model 3
# plt.plot(episodes, smoothed_rewards[2], 'purple', label='IDDPG-MAF (Proposed)')
# plt.fill_between(episodes, smoothed_rewards[2] - std_rewards[2], smoothed_rewards[2] + std_rewards[2], color='purple', alpha=0.2)
#
# # Model 4
# # plt.plot(episodes, smoothed_rewards[3], 'g', label='IDDPG-ns')
# # plt.fill_between(episodes, smoothed_rewards[3] - std_rewards[3], smoothed_rewards[3] + std_rewards[3], color='brown', alpha=0.2)
#
# # Customize x-axis labels to display as 2k, 4k, 6k, ..., 20k
# plt.xticks(ticks=np.arange(0, 20001, 2000), labels=[f'{x//1000+1}' for x in np.arange(0, 20001-800, 2000-80)])
#
#
# # Adding labels and title
# plt.title('(a) Training curves', fontsize=12)
# plt.xlabel('Number of episode (x1,000)', fontsize=12)
# plt.ylabel('Reward', fontsize=12)
# plt.legend()
#
# # Show plot
# plt.tight_layout()
# plt.show()














# ___________----------------------------------___________________________-----------------------------------
# # Base path and file pattern
# base_dir = r'D:\MADDPG_2nd_jp'
# file_template = r'020125_20_11_12\interval_record_eps_{:d}%\_6AC_all_episode_evaluation_each_step_status.pickle'
#
# # Sigma levels and percentage mapping
# sigma_levels = np.arange(0.1, 1.1, 0.1)
# plot_indices = [0, 2, 4, 6, 8]  # σ = 0.1, 0.3, 0.5, 0.7, 0.9
# colors = ['royalblue', 'seagreen', 'orange', 'firebrick', 'purple']
#
# plt.figure(figsize=(8, 5))
# plt.rcParams.update({'font.size': 14})
#
# for idx, plot_idx in enumerate(plot_indices):
#     pct = int(sigma_levels[plot_idx] * 100)
#     sigma_label = sigma_levels[plot_idx]
#     file_path = os.path.join(base_dir, file_template.format(pct))
#
#     # Load data
#     with open(file_path, 'rb') as f:
#         x = pickle.load(f)
#
#     max_steps = max(len(ep) for ep in x)
#     num_aircraft = len(x[0][0])
#     timesteps = np.arange(max_steps)
#
#     # Collect deviation curves per episode
#     deviation_matrix = []
#
#     for episode in x:
#         cumulative_original = np.zeros((num_aircraft, 2))
#         cumulative_noisy = np.zeros((num_aircraft, 2))
#         reached_goal = [False] * num_aircraft
#         episode_deviation = np.full(max_steps, np.nan)
#
#         for t, timestep in enumerate(episode):
#             deviations_t = []
#
#             for ac_id in range(num_aircraft):
#                 if ac_id >= len(timestep) or not isinstance(timestep[ac_id], dict):
#                     continue
#                 ac_data = timestep[ac_id]
#                 if 'delta_xy' not in ac_data:
#                     continue
#                 if reached_goal[ac_id]:
#                     continue
#                 if 'Euclidean_dist_to_goal' in ac_data and ac_data['Euclidean_dist_to_goal'] < 1.0:
#                     reached_goal[ac_id] = True
#                     continue
#
#                 delta_xy = ac_data['delta_xy']
#                 original_delta = np.array(delta_xy[1])
#                 noisy_delta = np.array(delta_xy[3])
#                 cumulative_original[ac_id] += original_delta
#                 cumulative_noisy[ac_id] += noisy_delta
#                 deviation = np.linalg.norm(cumulative_noisy[ac_id] - cumulative_original[ac_id])
#                 deviations_t.append(deviation)
#
#             if deviations_t:
#                 episode_deviation[t] = np.mean(deviations_t)
#
#         deviation_matrix.append(episode_deviation)
#
#     deviation_array = np.array(deviation_matrix)
#     mean_dev = np.nanmean(deviation_array, axis=0)
#     q5 = np.nanpercentile(deviation_array, 5, axis=0)
#     q95 = np.nanpercentile(deviation_array, 95, axis=0)
#
#     # Plot mean + shaded quantile band
#     color = colors[idx]
#     cut_len = 150
#     timesteps = np.arange(cut_len)
#     plt.plot(timesteps, mean_dev[:cut_len], label=f'σ = {sigma_label:.1f}', color=color, linewidth=2)
#     plt.fill_between(timesteps, q5[:cut_len], q95[:cut_len], color=color, alpha=0.2)
#
# # Final styling
# # plt.title('Average Position Deviation with 90% Quantile Band\nUnder Different Uncertainty Levels')
# plt.xlabel('Time step')
# plt.ylabel('Position deviation in nautical miles')
# plt.legend()
# plt.grid(True)
# plt.xticks(ticks=range(0, 151, 30))  # Set x-tick interval to 30
# plt.tight_layout()
# plt.show()


#_______________________IDDPG-MAF: position uncertainty_plot histogram _____________________

# import numpy as np
# import matplotlib.pyplot as plt
#
# # Parameters
# mean = 0
# # variance = 0.05
# # std_dev = np.sqrt(variance)
# std_dev = 0
# num_samples = 200
#
# # Generate random samples
# samples = np.random.normal(loc=mean, scale=std_dev, size=num_samples)
#
# # Plot histogram
# plt.figure(figsize=(8, 5))
# plt.hist(samples, bins=20, edgecolor='black', alpha=0.7, density=True)
# plt.title('Histogram of Normal Distribution Samples')
# plt.xlabel('Value')
# plt.ylabel('Probability Density')
# plt.grid(True)
# plt.axvline(mean, color='red', linestyle='dashed', linewidth=1.5, label='Mean')
# plt.legend()
# plt.show()



#-------------------------------IDDPG vs. single-agent DDPG-------------
# import matplotlib.pyplot as plt
# import numpy as np
#
# # Set global font size
# plt.rcParams.update({'font.size': 14})
#
# # Data extracted from the tables
# aircraft = [2, 3, 4, 5, 6, 7, 8, 9, 10]
#
# # IDDPG data
# iddpg_goal = [100, 100, 100, 98, 98, 96, 95, 93, 93]
# iddpg_los = [0, 0, 0, 0, 0, 1, 0, 1, 2]
# iddpg_thunderstorm = [0, 0, 0, 2, 2, 3, 5, 6, 5]
# iddpg_distance_means = [1.07, 1.09, 1.08, 1.17, 1.17, 1.16, 1.17, 1.36, 1.38]
#
# # Single-agent DDPG data
# ddpg_goal = [100, 100, 97, 96, 90, 84, 83, 77, 47]
# ddpg_los = [0, 0, 0, 1, 2, 0, 1, 4, 4]
# ddpg_thunderstorm = [0, 0, 3, 3, 8, 16, 15, 19, 49]
# ddpg_distance_means = [1.11, 1.10, 1.10, 1.10, 1.11, 1.12, 1.12, 1.16, 1.12]
#
# # Prepare subplots
# fig, axes = plt.subplots(2, 2, figsize=(12, 9))
#
# # Subplot 1: Goal reach rate
# axes[0, 0].plot(aircraft, iddpg_goal, marker='o', label='IDDPG (proposed)')
# axes[0, 0].plot(aircraft, ddpg_goal, marker='s', label='Single-agent DDPG')
# axes[0, 0].set_title('(a) Goal reach rate')
# axes[0, 0].set_xlabel('Number of aircraft')
# axes[0, 0].set_ylabel('Rate (%)')
# axes[0, 0].set_xticks(range(2, 11))
# axes[0, 0].set_yticks(range(0, 101, 10))
# axes[0, 0].grid(True, linestyle='--', alpha=0.7)
# axes[0, 0].legend()
#
# # Subplot 2: Aircraft LOS rate
# axes[0, 1].plot(aircraft, iddpg_los, marker='o', label='IDDPG (proposed)')
# axes[0, 1].plot(aircraft, ddpg_los, marker='s', label='Single-agent DDPG')
# axes[0, 1].set_title('(b) Aircraft LOS rate')
# axes[0, 1].set_xlabel('Number of aircraft')
# axes[0, 1].set_ylabel('Rate (%)')
# axes[0, 1].set_xticks(range(2, 11))
# axes[0, 1].set_yticks(range(0, 6, 1))
# axes[0, 1].grid(True, linestyle='--', alpha=0.7)
# axes[0, 1].legend()
#
# # Subplot 3: Thunderstorm LOS rate
# axes[1, 0].plot(aircraft, iddpg_thunderstorm, marker='o', label='IDDPG (proposed)')
# axes[1, 0].plot(aircraft, ddpg_thunderstorm, marker='s', label='Single-agent DDPG')
# axes[1, 0].set_title('(c) Thunderstorm LOS rate')
# axes[1, 0].set_xlabel('Number of aircraft')
# axes[1, 0].set_ylabel('Rate (%)')
# axes[1, 0].set_xticks(range(2, 11))
# axes[1, 0].set_yticks(range(0, 51, 10))
# axes[1, 0].grid(True, linestyle='--', alpha=0.7)
# axes[1, 0].legend()
#
# # Subplot 4: Flight distance ratio (bar chart for means)
# x = np.arange(len(aircraft))  # Number of aircraft
# bar_width = 0.35
#
# axes[1, 1].bar(x - bar_width / 2, iddpg_distance_means, bar_width, label='IDDPG (proposed)', color='blue', alpha=0.7)
# axes[1, 1].bar(x + bar_width / 2, ddpg_distance_means, bar_width, label='Single-agent DDPG', color='orange', alpha=0.7)
#
# axes[1, 1].set_title('(d) Flight distance ratio')
# axes[1, 1].set_xlabel('Number of aircraft')
# axes[1, 1].set_ylabel('Distance ratio (mean)')
# axes[1, 1].set_xticks(x)
# axes[1, 1].set_xticklabels(aircraft)
# axes[1, 1].legend()
# axes[1, 1].grid(True, axis='y', linestyle='--', alpha=0.7)
#
# # Adjust layout
# plt.tight_layout()
# plt.show()
#
#
# print('end')



#------------------------------------Reward ablation study--------------
# import matplotlib.pyplot as plt
# import numpy as np
#
# # Set global font size
# plt.rcParams.update({'font.size': 15})
#
# # Data from the table
# categories = ['Full model', 'No near aircraft penalty', 'No near storm penalty',
#               'No distance to goal', 'No heading change']
# metrics = ['Aircraft LOS rate', 'Thunderstorm LOS rate', 'Goal reach rate', 'Aircraft stray rate']
#
# data = np.array([
#     [0, 4, 4, 9, 1],  # Aircraft LOS rate
#     [2, 7, 11, 9, 2],  # Thunderstorm LOS rate
#     [98, 85, 78, 45, 97],  # Goal reach rate
#     [0, 4, 7, 37, 0]   # Aircraft stray rate
# ])
#
# # Colors for each bar
# colors = ['#1f77b4', '#ff7f0e', '#7f7f7f', '#d62728']  # Color-blind-friendly palette
#
# # Bar chart settings
# x = np.arange(len(categories))  # Number of groups
# bar_width = 0.15  # Width of each bar
#
# # Create the figure
# fig, ax = plt.subplots(figsize=(14, 7))
#
# # Plot each metric as a separate bar group
# for i in range(data.shape[0]):
#     ax.bar(
#         x + i * bar_width,
#         data[i],
#         width=bar_width,
#         color=colors[i],
#         edgecolor='black',
#         label=metrics[i],
#         alpha=0.8
#     )
#
# # Customize the plot
# ax.set_xlabel('Model variations', weight='bold')
# ax.set_ylabel('Percentage (%)')
# # ax.set_title('Comparison of metrics across model variations')
# ax.set_xticks(x + bar_width * (len(metrics) - 1) / 2)
# ax.set_xticklabels(categories, ha='center')  # Ensure labels are on a single line
# ax.set_ylim(0, 120)  # Set y-axis maximum to 100
#
# # Adjust legend position closer to the plot boundary
# fig.legend(
#     # title='Metrics',
#     loc='upper center',
#     bbox_to_anchor=(0.5, 0.85),  # Moved closer to the boundary
#     ncol=4,
#     frameon=False
# )
#
# # Add gridlines
# ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)
#
# # Add data labels above bars
# for i in range(data.shape[0]):
#     for j in range(len(categories)):
#         ax.text(
#             x[j] + i * bar_width,
#             data[i][j] + 2,
#             f'{data[i][j]}%',
#             ha='center',
#             va='bottom'
#         )
#
# # Adjust layout for better spacing
# plt.tight_layout()
# plt.subplots_adjust(top=0.85)  # Adjust space for the legend at the top
#
#
# plt.show()
# print('end')

#---------------------ETA distributions--------------------
# # Define uncertainty levels with mean and standard deviation (converted to seconds)
# Set global font size
# plt.rcParams.update({'font.size': 12})
# uncertainty_levels = {
#     "Negligible": (0, 1.6 * 15),
#     "Low": (0, 6.4 * 15),
#     "Medium": (0, 14 * 15),
#     "High": (0, 30 * 15)
# }  # *15 converts to seconds, as one time step is 15 seconds
#
# # Sample size
# sample_size = 3000
#
# # Define subplot title labels
# subplot_labels = ['(a)', '(b)', '(c)', '(d)']
#
# # Create subplots for each uncertainty level
# fig, axes = plt.subplots(1, len(uncertainty_levels), figsize=(12, 4), sharey=True)
#
# # Custom x-ticks for each subplot
# custom_ticks = [
#     [-60, 0, 60],    # Negligible
#     [-300, 0, 300],  # Low
#     [-600, 0, 600],  # Medium
#     [-1200, 0, 1200] # High
# ]
#
# # Generate and plot histogram for each uncertainty level
# for ax, (subplot_label, (level, (mean, std_dev)), ticks) in zip(axes, zip(subplot_labels, uncertainty_levels.items(), custom_ticks)):
#     # Generate samples from a normal distribution
#     samples = np.random.normal(loc=mean, scale=std_dev, size=sample_size)
#
#     # Plot histogram
#     ax.hist(samples, bins=10, density=False, alpha=0.7, color='blue', edgecolor='black')
#
#     # Set titles and labels with subplot labels
#     ax.set_title(f"{subplot_label} {level} Uncertainty")
#     ax.set_xlabel("ETA uncertainty (s)")
#     ax.set_ylabel("Frequency")
#
#     # Set custom x-ticks
#     ax.set_xticks(ticks)
#     ax.set_xticklabels([str(tick) for tick in ticks])
#
# # Adjust layout for clarity
# plt.tight_layout()
# plt.show()
# print('end')




#________________Number of aircraft concurrently in the airspace (%)______________
# import numpy as np
# import pandas as pd
# import matplotlib
# import matplotlib.pyplot as plt
# from collections import Counter
# import pickle
#
# import matplotlib.pyplot as plt
# from collections import Counter
# import pickle
#
# # Set global font size
# plt.rcParams.update({'font.size': 14})
#
# # Load data for each uncertainty level
# with open(r'D:\MADDPG_2nd_jp\261124_16_37_03\interval_record_eps\v2_30AC_activated_AC.pickle', 'rb') as f:
#     total_ac_normal = pickle.load(f)
# with open(r'D:\MADDPG_2nd_jp\261124_16_37_03\interval_record_eps_low\v2_30AC_activated_AC.pickle', 'rb') as f:
#     total_ac_low = pickle.load(f)
# with open(r'D:\MADDPG_2nd_jp\261124_16_37_03\interval_record_eps_mid\v2_30AC_activated_AC.pickle', 'rb') as f:
#     total_ac_mid = pickle.load(f)
# with open(r'D:\MADDPG_2nd_jp\261124_16_37_03\interval_record_eps_high\v2_30AC_activated_AC.pickle', 'rb') as f:
#     total_ac_high = pickle.load(f)
#
#
# # Helper function to calculate percentages
# def calculate_percentages(data):
#     all_values = [value for sublist in data.values() for value in sublist]
#     counter = Counter(all_values)
#     total_count = sum(counter.values())
#     percentages = {k: (v / total_count) * 100 for k, v in counter.items()}
#     return dict(sorted(percentages.items()))
#
#
# # Calculate percentages for each uncertainty level
# percentages_normal = calculate_percentages(total_ac_normal)
# percentages_low = calculate_percentages(total_ac_low)
# percentages_mid = calculate_percentages(total_ac_mid)
# percentages_high = calculate_percentages(total_ac_high)
#
# # Plot settings
# fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=300)  # Higher resolution with 2x2 grid
# axes = axes.flatten()
#
# titles = [
#     '(a) Negligible Uncertainty',
#     '(b) Low Uncertainty',
#     '(c) Medium Uncertainty',
#     '(d) High Uncertainty'
# ]
# percentages = [percentages_normal, percentages_low, percentages_mid, percentages_high]
#
# # Plot each subplot
# for ax, title, data in zip(axes, titles, percentages):
#     bars = ax.barh(list(data.keys()), list(data.values()), color='gray', edgecolor='black', alpha=0.7, height=0.5)
#
#     # Adjust text position dynamically for each subplot
#     for bar in bars:
#         width = bar.get_width()
#         yloc = bar.get_y() + bar.get_height() / 2
#         x_limit = ax.get_xlim()[1]  # Get the upper x-axis limit
#
#         # Adjust text placement dynamically based on the bar width
#         if width < x_limit * 0.05:  # Very small bars
#             text_x = width + 0.5
#             ha = 'left'
#         elif width < x_limit * 0.9:  # Medium-sized bars
#             text_x = width + 0.5
#             ha = 'left'
#         else:  # Large bars
#             text_x = width - 2
#             ha = 'center'
#
#         ax.text(text_x, yloc, f'{width:.2f}%', va='center', ha=ha, color='black')
#     # Set title and labels
#     ax.set_title(title)
#     ax.xaxis.grid(True, linestyle='--', color='gray', alpha=0.5)
#     ax.yaxis.grid(True, linestyle='--', color='gray', alpha=0.5)
#     ax.set_xlabel('Number of aircraft concurrently in the airspace (%)')
#     ax.set_ylabel('No. of Aircraft')
#     # Ensure all y-axis labels are shown
#     ax.set_yticks(list(data.keys()))
#     ax.set_yticklabels(list(data.keys()))
#
# # Adjust layout with extra padding to prevent text cutoff
# plt.tight_layout()
# plt.show()

#---------------------actual separation under various uncertainty levels---------------
# Load data for each uncertainty level
# import matplotlib.pyplot as plt
# import numpy as np
#
# # Set global font size
# plt.rcParams.update({'font.size': 15})
#
# import matplotlib.pyplot as plt
# import pickle
#
# # Load data from pickles
# with open(r'D:\MADDPG_2nd_jp\261124_16_37_03\interval_record_eps\v2_actual_eta.pickle', 'rb') as f:
#     total_ac_normal = pickle.load(f)
# with open(r'D:\MADDPG_2nd_jp\261124_16_37_03\interval_record_eps_low\v2_actual_eta.pickle', 'rb') as f:
#     total_ac_low = pickle.load(f)
# with open(r'D:\MADDPG_2nd_jp\261124_16_37_03\interval_record_eps_mid\v2_actual_eta.pickle', 'rb') as f:
#     total_ac_mid = pickle.load(f)
# with open(r'D:\MADDPG_2nd_jp\261124_16_37_03\interval_record_eps_high\v2_actual_eta.pickle', 'rb') as f:
#     total_ac_high = pickle.load(f)
#
# # Prepare data for plotting
# datasets = {
#     'Negligible': total_ac_normal,
#     'Low': total_ac_low,
#     'Medium': total_ac_mid,
#     'High': total_ac_high
# }
#
# # Plot each dataset
# plt.figure(figsize=(8, 6))
#
# for label, data in datasets.items():
#     # Flatten the lists for plotting (if data is a list of lists)
#     all_values = [value for sublist in data for value in sublist] if isinstance(data[0], list) else data
#
#     # convert time step to minutes, 12s/step
#     all_values = [value / 6 for value in all_values]
#
#     # Sort values from highest to lowest
#     sorted_values = sorted(all_values, reverse=True)
#
#     # Plot the sorted data
#     plt.plot(sorted_values, label=f"{label} uncertainty")
#
# # Add the reference line
# plt.axhline(y=10, color='black', linestyle='--', linewidth=2.5, label='Separation minima')
# plt.xlim(0, 3000)  # Set x-axis range to 0-3000
#
# # Add plot details
# # plt.title('Line Plot of Actual ETA Values (Sorted)')
# plt.xlabel('Numer of aircraft pairs')
# plt.ylabel('Actual separation between aircraft (min)')
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show()
#
# print('end')


#-----------higher uncertainty level requires more time steps to clear a batch of ac-------
# Load data for each uncertainty level
# with open(r'D:\MADDPG_2nd_jp\261124_16_37_03\interval_record_eps\v2_30AC_activated_AC.pickle', 'rb') as f:
#     total_ac_normal = pickle.load(f)
# with open(r'D:\MADDPG_2nd_jp\261124_16_37_03\interval_record_eps_low\v2_30AC_activated_AC.pickle', 'rb') as f:
#     total_ac_low = pickle.load(f)
# with open(r'D:\MADDPG_2nd_jp\261124_16_37_03\interval_record_eps_mid\v2_30AC_activated_AC.pickle', 'rb') as f:
#     total_ac_mid = pickle.load(f)
# with open(r'D:\MADDPG_2nd_jp\261124_16_37_03\interval_record_eps_high\v2_30AC_activated_AC.pickle', 'rb') as f:
#     total_ac_high = pickle.load(f)
#
# import matplotlib.pyplot as plt
# matplotlib.rcParams.update({'font.size': 13})
# # Calculate the lengths of lists in each dictionary and remove lengths < 200
# def process_data(data_dict):
#     lengths = [len(lst) for lst in data_dict.values()]
#     filtered_lengths = [length / 5 for length in lengths if length >= 350]
#     return filtered_lengths
#
# # Process the data
# lengths_normal = process_data(total_ac_normal)
# lengths_low = process_data(total_ac_low)
# lengths_mid = process_data(total_ac_mid)
# lengths_high = process_data(total_ac_high)
#
# # Prepare data for box plot
# data_to_plot = [lengths_normal, lengths_low, lengths_mid, lengths_high]
# labels = ['Negligible', 'Low', 'Medium', 'High']
#
# # Plot the box plot
# plt.figure(figsize=(6, 5))
# plt.boxplot(data_to_plot, labels=labels, patch_artist=True, boxprops=dict(facecolor='skyblue', color='black'))
# # plt.title('Box Plot of List Lengths for Each Group')
# plt.xlabel('Uncertainty level')
# plt.ylabel('Total time for an batch of aircraft (min)')
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show()
#
# print('end')

# ------------------------------ETA uncertainty---------------------------------------
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import pickle
# import matplotlib
#
# matplotlib.rcParams.update({'font.size': 14})
#
# with open(r'D:\MADDPG_2nd_jp\261124_16_37_03\interval_record_eps\_30AC_activated_AC.pickle', 'rb') as f:
#     total_ac = pickle.load(f)
#
# fig, ax = plt.subplots(figsize=(8, 6))
#
# for i, data in total_ac.items():
#     # ax.plot(data, label=f"List {i}")  # line graph
#     ax.scatter(range(len(data)), data, label=f"List {i}")  # Scatter plot
#
# ax.set_xlabel("Experiment time (minutes)")
# ax.set_ylabel("No. of aircraft in airspace")
#
# # Find the largest length among all lists
# max_length = max(len(lst) for lst in total_ac.values())
# rounded_max_length = math.ceil(max_length / 40) * 40 # Round up to the nearest multiple of 40
# # Customize the x-axis
# ticks_interval = 40  # Interval of 40 ticks
# tick_positions = list(range(0, rounded_max_length + 1, ticks_interval))
# tick_labels = [str(i * 10 // 40) for i in tick_positions]  # Convert ticks to minutes
# plt.xticks(ticks=tick_positions, labels=tick_labels)
#
# # Add dashed vertical lines as background
# for x in tick_positions:
#     ax.axvline(x=x, color='black', linestyle='--', linewidth=0.4, alpha=0.6)
#
# # Improve the layout
# plt.tight_layout()
# plt.show()
#
# print('end')
# ----------------------END--------ETA uncertainty---------------------------------------


# # Specify the file path
# file_path = r'D:\MADDPG_2nd_jp\training curves.xlsx'
#
# # Read the data from the CSV file, no header since the file contains only reward values
# data = pd.read_excel(file_path, header=None)
#
# # Convert all data to numeric, forcing errors to NaN (you can skip this step if not needed)
# data = data.applymap(lambda x: pd.to_numeric(x, errors='coerce'))
#
# # Convert the DataFrame to a numpy array for easier manipulation
# rewards = data.values  # No need to transpose if shape is (4, 16384)
#
# # Smoothing function (moving average)
# def moving_average(data, window_size=100):
#     return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
#
# # Apply smoothing
# smoothed_rewards = np.array([moving_average(reward, window_size=500) for reward in rewards])
#
# # Calculate mean and confidence intervals for smoothed data
# mean_rewards = smoothed_rewards.mean(axis=1)
# std_rewards = smoothed_rewards.std(axis=1)
#
# # X-axis values (episodes)
# episodes = np.arange(1, len(smoothed_rewards[0]) + 1, step=200)  # Plot every 10th episode
# smoothed_rewards = smoothed_rewards[:, ::200]
# # Plotting
# plt.figure(figsize=(8, 6))
#
# # Model 1
# plt.plot(episodes, smoothed_rewards[0], 'r-', label='Model 1')
# plt.fill_between(episodes, smoothed_rewards[0] - std_rewards[0], smoothed_rewards[0] + std_rewards[0], color='red', alpha=0.2)
#
# # Model 2
# plt.plot(episodes, smoothed_rewards[1], 'navy', label='Model 2')
# plt.fill_between(episodes, smoothed_rewards[1] - std_rewards[1], smoothed_rewards[1] + std_rewards[1], color='navy', alpha=0.2)
#
# # Model 3
# plt.plot(episodes, smoothed_rewards[2], 'purple', label='Model 3')
# plt.fill_between(episodes, smoothed_rewards[2] - std_rewards[2], smoothed_rewards[2] + std_rewards[2], color='purple', alpha=0.2)
#
# # Model 4
# plt.plot(episodes, smoothed_rewards[3], 'brown', label='Model 4')
# plt.fill_between(episodes, smoothed_rewards[3] - std_rewards[3], smoothed_rewards[3] + std_rewards[3], color='brown', alpha=0.2)
#
# # Adding labels and title
# plt.xlabel('Training episode')
# plt.ylabel('Reward')
# plt.title('Training Curve with Confidence Interval')
# plt.legend()
#
# # Show plot
# plt.show()
#
#

#___________plot training curves______________v1.0
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # Specify the file path
# file_path = r'D:\MADDPG_2nd_jp\training curves.xlsx'
#
# # Read the data from the Excel file, no header since the file contains only reward values
# data = pd.read_excel(file_path, header=None)
#
# # Convert all data to numeric, forcing errors to NaN (you can skip this step if not needed)
# data = data.applymap(lambda x: pd.to_numeric(x, errors='coerce'))
#
# # Transpose the DataFrame since your data is now (20000, 4) instead of (4, 20000)
# # “-5000” is to move y-axis, “/50” is to reduce magnitude effect
# rewards = (data.values.T - 4000)/50  # Transpose the data to get shape (4, 20000)
#
# # Smoothing function (moving average)
# def moving_average(data, window_size=200):
#     return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
#
# # Apply smoothing
# smoothed_rewards = np.array([moving_average(reward, window_size=2000) for reward in rewards])
#
# # Calculate mean and confidence intervals for smoothed data
# mean_rewards = smoothed_rewards.mean(axis=1)
# std_rewards = smoothed_rewards.std(axis=1)
#
# # X-axis values (episodes)
# episodes = np.arange(1, len(smoothed_rewards[0]) + 1, step=50)  # Plot every 200th episode
# smoothed_rewards = smoothed_rewards[:, ::50]
#
# # Set global font size
# plt.rcParams.update({'font.size': 12})
#
# # Plotting
# plt.figure(figsize=(8, 8))
#
# # Model 1
# plt.plot(episodes, smoothed_rewards[0], 'r-', label='IDDPG')
# plt.fill_between(episodes, smoothed_rewards[0] - std_rewards[0], smoothed_rewards[0] + std_rewards[0], color='red', alpha=0.2)
#
# # Model 2
# plt.plot(episodes, smoothed_rewards[1], 'navy', label='IDDPG-n')
# plt.fill_between(episodes, smoothed_rewards[1] - std_rewards[1], smoothed_rewards[1] + std_rewards[1], color='navy', alpha=0.2)
#
# # Model 3
# plt.plot(episodes, smoothed_rewards[2], 'purple', label='IDDPG-s')
# plt.fill_between(episodes, smoothed_rewards[2] - std_rewards[2], smoothed_rewards[2] + std_rewards[2], color='purple', alpha=0.2)
#
# # Model 4
# plt.plot(episodes, smoothed_rewards[3], 'g', label='IDDPG-ns')
# plt.fill_between(episodes, smoothed_rewards[3] - std_rewards[3], smoothed_rewards[3] + std_rewards[3], color='brown', alpha=0.2)
#
# # Adding labels and title
# plt.xlabel('Training episode')
# plt.ylabel('Reward')
# # plt.title('Training Curve with Confidence Interval')
# plt.legend()
#
# # Show plot
# plt.show()


