import sys
# sys.path.append('F:\githubClone\Multi_agent_AAC\old_framework_test')
# sys.path.append('D:\Multi_agent_AAC\old_framework_test')
from openpyxl import load_workbook
from openpyxl import Workbook
import argparse
import datetime
import pandas as pd
import numpy as np
import torch
import os
import time
import matplotlib.animation as animation
import pickle
import wandb
from pc_specific import PCconfig
from maddpg_agent_randomOD_radar_sur_drones_N_Model_use_tdCPA_forV2 import MADDPG
from utils_randomOD_radar_sur_drones_N_Model_use_tdCPA_forV2 import *
from grid_env_generation_newframe_randomOD_radar_sur_drones_N_Model_use_tdCPA_forV2 import env_generation
from env_simulator_randomOD_radar_sur_drones_N_Model_use_tdCPA_forV2 import env_simulator
from copy import deepcopy
import torch
import matplotlib.pyplot as plt
import matplotlib
from shapely.geometry import LineString, Point, Polygon
from shapely.strtree import STRtree
from matplotlib.markers import MarkerStyle
import math
from matplotlib.transforms import Affine2D
from Utilities_own_randomOD_radar_sur_drones_N_Model_use_tdCPA_forV2 import *
from collections import deque
import csv

num_devices = torch.cuda.device_count()
print("Number of GPUs:", num_devices)
# Get the names of the available GPUs
gpu_names = [torch.cuda.get_device_name(i) for i in range(num_devices)]
print("GPU Names:", gpu_names)
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using GPU')
else:
    device = torch.device('cpu')
    print('Using CPU')

# device = torch.device('cpu')  # Desktop, we must use this.


def main(args):
    if args.mode == "train":
        today = datetime.date.today()
        current_date = today.strftime("%d%m%y")
        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime("%H_%M_%S")
        file_name = 'D:\MADDPG_2nd_jp/' + str(current_date) + '_' + str(formatted_time)
        if not os.path.exists(file_name):
            os.makedirs(file_name)
        plot_file_name = file_name + '/toplot'
        if not os.path.exists(plot_file_name):
            os.makedirs(plot_file_name)
        # ------------ this portion is to save using excel instead of pickle -----------
        # excel_file_path_reward = plot_file_name + '/all_episode_reward.xlsx'
        # excel_file_path_noise = plot_file_name + '/all_episode_noise.xlsx'
        # excel_file_path_time = plot_file_name + '/all_episode_time.xlsx'
        # initialize_excel_file(excel_file_path_reward)
        # initialize_excel_file(excel_file_path_noise)
        # initialize_excel_file(excel_file_path_time)
        # ------------ end of this portion is to save using excel instead of pickle -----------

    use_wanDB = False
    # use_wanDB = True

    # evaluation_by_episode = True
    evaluation_by_episode = False

    # get_evaluation_status = True  # have figure output
    get_evaluation_status = False  # no figure output, mainly obtain collision rate

    # simply_view_evaluation = True  # don't save gif
    simply_view_evaluation = False  # save gif

    # full_observable_critic_flag = True
    full_observable_critic_flag = False

    # transfer_learning = True
    transfer_learning = False

    # use_GRU_flag = True
    use_GRU_flag = False

    # use_single_portion_selfATT = True # Neural network config, this config only apply to one portion state space, 6+nx5
    use_single_portion_selfATT = False

    # use_selfATT_with_radar = True
    use_selfATT_with_radar = False

    use_allNeigh_wRadar = True
    # use_allNeigh_wRadar = False

    if use_allNeigh_wRadar:
        # own_obs_only = True
        own_obs_only = False
    else:
        own_obs_only = False

    if use_wanDB:
        wandb.login(key="efb76db851374f93228250eda60639c70a93d1ec")
        wandb.init(
            # set the wandb project where this run will be logged
            project="MADDPG_FrameWork",
            name='MADDPG_test_'+str(current_date) + '_' + str(formatted_time),
            # track hyperparameters and run metadata
            config={
                # "learning_rate": args.a_lr,
                "epochs": args.max_episodes,
            }
        )

    # -------------- create my own environment -----------------
    # set boundary
    xlow = 455
    xhigh = 680
    ylow = 255
    yhigh = 385
    bound = [xlow, xhigh, ylow, yhigh]
    # generate static env from shape file
    shapePath = PCconfig().shape_path
    # shapePath = 'D:\deep_Q_learning\DQN_new_framework\lakesideMap\lakeSide.shp'
    # shapePath = 'F:\githubClone\deep_Q_learning\DQN_new_framework\lakesideMap\lakeSide.shp'
    # shapePath = 'D:\github_clone\Multi_agent_AAC\MA_ver1\lakesideMap\lakeSide.shp'
    staticEnv = env_generation(shapePath, bound)
    max_xy = staticEnv[-1]
    env = env_simulator(staticEnv[0], staticEnv[1], staticEnv[2], bound, staticEnv[3])
    # total_agentNum = len(pd.read_excel(env.agentConfig))
    # total_agentNum = 3
    # total_agentNum = 5
    total_agentNum = 8
    # total_agentNum = 1
    # max_nei_num = 5
    # create world
    # actor_dim = [6+(total_agentNum-1)*2, 10, 6]  # dim host, maximum dim grid, dim other drones
    # critic_dim = [6+(total_agentNum-1)*2, 10, 6]
    if full_observable_critic_flag:
        # actor_dim = [6, 18, 6]  # dim host, maximum dim grid, dim other drones
        # actor_dim = [8, 18, 6]  # dim host, maximum dim grid, dim other drones
        # actor_dim = [9, (total_agentNum - 1) * 8, 36, 6]  # dim host, maximum dim grid, dim other drones
        actor_dim = [7, (total_agentNum - 1) * 5, 18, 6]  # dim host, maximum dim grid, dim other drones
        # actor_dim = [26, 18, 6]  # dim host, maximum dim grid, dim other drones
        # critic_dim = [6, 18, 6]
        # critic_dim = [8, 18, 6]
        # critic_dim = [total_agentNum*9, total_agentNum*36, 6]
        # critic_dim = [total_agentNum*(7+5), total_agentNum*18, 6]
        critic_dim = [7, (total_agentNum - 1) * 5, 18, 6]
        # critic_dim = [7]
        # critic_dim = [26, 18, 6]
        # critic_dim = [ea_dim * total_agentNum for ea_dim in actor_dim]
    else:
        if use_selfATT_with_radar:
            # actor_dim = [6, (total_agentNum - 1) * 5, 18, 6]
            actor_dim = [6, (total_agentNum - 1) * 5, 18, 6]
            # critic_dim = [6, (total_agentNum - 1) * 5, 18, 6]
            critic_dim = [6, (total_agentNum - 1) * 5, 18, 6]
        elif use_allNeigh_wRadar:
            # actor_dim = [6, (total_agentNum - 1) * 5, 18, 6]
            # actor_dim = [7, (total_agentNum - 1) * 6, 36, 6]
            # actor_dim = [9, (total_agentNum - 1) * 8, 36, 6]
            # actor_dim = [9, (total_agentNum - 1) * 8, 18, 6]
            actor_dim = [7, (total_agentNum - 1) * 5, 18, 6]
            # actor_dim = [9, (total_agentNum - 1) * 5, 18, 6]
            # actor_dim = [6, 1 * 5, 36, 6]
            # actor_dim = [6, 2 * 5, 36, 6]
            # critic_dim = [6, (total_agentNum - 1) * 5, 18, 6]
            # critic_dim = [7, (total_agentNum - 1) * 6, 36, 6]
            # critic_dim = [9, (total_agentNum - 1) * 8, 36, 6]
            # critic_dim = [9, (total_agentNum - 1) * 8, 18, 6]
            critic_dim = [7, (total_agentNum - 1) * 5, 18, 6]
            # critic_dim = [9, (total_agentNum - 1) * 5, 18, 6]
            # critic_dim = [6, 1 * 5, 36, 6]
            # critic_dim = [6, 2 * 5, 36, 6]
        else:
            # actor_dim = [6, 18, 6]  # dim host, maximum dim grid, dim other drones
            # actor_dim = [8, 18, 6]  # dim host, maximum dim grid, dim other drones
            # actor_dim = [10, 18, 6]  # dim host, maximum dim grid, dim other drones
            # actor_dim = [18, 18, 6]  # dim host, maximum dim grid, dim other drones
            # actor_dim = [16, 18, 6]  # dim host, maximum dim grid, dim other drones
            # actor_dim = [8+(total_agentNum-1)*4, 18, 6]  # dim host, maximum dim grid, dim other drones
            # actor_dim = [6+(total_agentNum-1)*5, 18, 6]  # dim host, maximum dim grid, dim other drones
            # actor_dim = [6+(total_agentNum-1)*4, (total_agentNum-1)*1, 6]  # dim host, maximum dim grid, dim other drones
            actor_dim = [6, (total_agentNum-1)*5, 6]  # dim host, maximum dim grid, dim other drones
            # actor_dim = [14, 18, 6]  # dim host, maximum dim grid, dim other drones
            # actor_dim = [11, 18, 6]  # dim host, maximum dim grid, dim other drones
            # actor_dim = [12, 18, 6]  # dim host, maximum dim grid, dim other drones
            # actor_dim = [26, 18, 6]
            # critic_dim = [6, 18, 6]
            # critic_dim = [8, 18, 6]
            # critic_dim = [10, 18, 6]
            # critic_dim = [18, 18, 6]
            # critic_dim = [16, 18, 6]
            # critic_dim = [8+(total_agentNum-1)*4, 18, 6]  # dim host, maximum dim grid, dim other drones
            # critic_dim = [6+(total_agentNum-1)*5, 18, 6]  # dim host, maximum dim grid, dim other drones
            # critic_dim = [6+(total_agentNum-1)*4, (total_agentNum-1)*1, 6]  # dim host, maximum dim grid, dim other drones
            critic_dim = [6, (total_agentNum-1)*5, 6]  # dim host, maximum dim grid, dim other drones
            # critic_dim = [14, 18, 6]
            # critic_dim = [11, 18, 6]
            # critic_dim = [12, 18, 6]
            # critic_dim = [26, 18, 6]

    actor_hidden_state = 64
    actor_hidden_state_list = [actor_hidden_state for _ in range(total_agentNum)]

    gru_history_length = 10
    gru_history = deque(maxlen=gru_history_length)
    args.gru_history_length = gru_history_length
    # critic_dim = [9, 9, 9]
    # critic_dim = [16, 9, 6]
    n_actions = 2
    acc_max = 8
    # acc_max = 30
    acc_range = [-acc_max, acc_max]  # NOTE this we need to change

    # actorNet_lr = 0.001/10
    # actorNet_lr = 0.0001/5
    # actorNet_lr = 0.0005
    # actorNet_lr = 0.001
    actorNet_lr = 0.0001
    # actorNet_lr = 0.0001/2
    # actorNet_lr = 0.001
    # criticNet_lr = 0.001/10
    # criticNet_lr = 0.0001/5
    # criticNet_lr = 0.0005
    # criticNet_lr = 0.001
    criticNet_lr = 0.0001
    # criticNet_lr = 0.0001/2
    # criticNet_lr = 0.001
    # criticNet_lr = 0.0005

    # noise parameter ini
    largest_Nsigma = 0.5
    smallest_Nsigma = 0.15
    ini_Nsigma = largest_Nsigma

    # max_spd = 15
    # max_spd = 10
    max_spd = 5
    env.create_world(total_agentNum, n_actions, args.gamma, args.tau, args.update_step, largest_Nsigma, smallest_Nsigma, ini_Nsigma, max_xy, max_spd, acc_range)

    # --------- my own -----------
    n_agents = len(env.all_agents)
    n_actions = n_actions

    torch.manual_seed(args.seed)  # this is the seed

    if args.algo == "maddpg":
        model = MADDPG(actor_dim, critic_dim, n_actions, actor_hidden_state, gru_history_length, n_agents, args, criticNet_lr, actorNet_lr, args.gamma, args.tau, full_observable_critic_flag, use_GRU_flag, use_single_portion_selfATT, use_selfATT_with_radar, use_allNeigh_wRadar, own_obs_only, env.normalizer, device)

    episode = 0
    current_row = 0
    excel_file_path = '../experience_replay_data.xlsx'
    writer = pd.ExcelWriter(excel_file_path, engine='xlsxwriter')
    total_step = 0
    score_history = []
    experience_replay_record = []
    eps_reward_record = []
    eps_check_collision = []
    eps_noise_record = []
    episode_critic_loss_cal_record = []
    # eps_end = 500  # at eps = eps_end, the eps value drops to lowest value which is 0.03 (this value is fixed)
    # eps_end = 5000  # at eps = eps_end, the eps value drops to lowest value which is 0.03 (this value is fixed)
    # eps_end = round(args.max_episodes / 2)  # at eps = eps_end, the eps value drops to lowest value which is 0.03 (this value is fixed)
    # eps_end = 8000  # at eps = eps_end, the eps value drops to lowest value which is 0.03 (this value is fixed)
    eps_end = 10000  # at eps = eps_end, the eps value drops to lowest value which is 0.03 (this value is fixed)
    # eps_end = 2500  # at eps = eps_end, the eps value drops to lowest value which is 0.03 (this value is fixed)
    # eps_end = 4500  # at eps = eps_end, the eps value drops to lowest value which is 0.03 (this value is fixed)
    # eps_end = 1000  # at eps = eps_end, the eps value drops to lowest value which is 0.03 (this value is fixed)
    # eps_end = 2000  # at eps = eps_end, the eps value drops to lowest value which is 0.03 (this value is fixed)
    noise_start_level = 1
    training_start_time = time.time()

    # ------------ record episode time ------------- #
    eps_time_record = []
    # ----------- record each collision checking version running time and decision -------#
    collision_count = 0
    one_drone_reach = 0
    two_drone_reach = 0
    three_drone_reach = 0
    four_drone_reach = 0
    five_drone_reach = 0
    six_drone_reach = 0
    seven_drone_reach = 0
    all_drone_reach = 0
    all_steps_used = 0
    sorties_reached = 0
    idle_drone = 0
    crash_to_bound = 0
    crash_to_building = 0
    crash_to_drone = 0
    crash_due_to_nearest = 0
    episode_goal_found = [False] * n_agents
    dummy_xy = (None, None)  # this is a dummy tuple of xy, is not useful during normal training, it is only useful when generating reward map
    if args.mode == "eval":
        # args.max_episodes = 10  # only evaluate one episode during evaluation mode.
        # args.max_episodes = 5  # only evaluate one episode during evaluation mode.
        args.max_episodes = 100
        # args.max_episodes = 1
        # args.max_episodes = 250
        # args.max_episodes = 25
        pre_fix = PCconfig().save_file_prefix
        # episode_to_check = str(10000)
        # pre_fix = r'F:\OneDrive_NTU_PhD\OneDrive - Nanyang Technological University\DDPG_2ndJournal\dim_8_transfer_learning'
        episode_to_check = str(20000)
        model_list = []
        if full_observable_critic_flag:
            for i in range(total_agentNum):
                load_filepath = pre_fix + '\episode_' + episode_to_check + '_' + str(i)+ '_actor_net.pth'
                model_list.append(load_filepath)
            model.load_model(model_list, full_observable_critic_flag)
        else:
            # using one model, so we load all the same
            load_filepath_0 = pre_fix + '\episode_' + episode_to_check + '_actor_net.pth'
            load_filepath_1 = pre_fix + '\episode_' + episode_to_check + '_actor_net.pth'
            load_filepath_2 = pre_fix + '\episode_' + episode_to_check + '_actor_net.pth'
            # load_filepath_3 = pre_fix + '\episode_' + episode_to_check + '_agent_3actor_net.pth'
            # load_filepath_4 = pre_fix + '\episode_' + episode_to_check + '_agent_4actor_net.pth'

            # model.load_model([load_filepath_0, load_filepath_1, load_filepath_2, load_filepath_3, load_filepath_4])
            model.load_model([load_filepath_0, load_filepath_1, load_filepath_2], full_observable_critic_flag)
    else:
        if transfer_learning:
            pre_fix = r'F:\OneDrive_NTU_PhD\OneDrive - Nanyang Technological University\DDPG_2ndJournal\dim_8_transfer_learning'
            episode_to_check = str(21000)
            load_filepath_0 = pre_fix + '\episode_' + episode_to_check + '_agent_0actor_net.pth'
            load_filepath_1 = pre_fix + '\episode_' + episode_to_check + '_agent_1actor_net.pth'
            load_filepath_2 = pre_fix + '\episode_' + episode_to_check + '_agent_2actor_net.pth'
            model.load_model([load_filepath_0, load_filepath_1, load_filepath_2])
            print("training start with transfer learning (pre-loaded actor model)")
    # while episode < args.max_episodes:
    steps_before_collide = []
    while episode < args.max_episodes:  # start of an episode

        # ------------ my own env.reset() ------------ #
        episode_start_time = time.time()
        episode += 1
        eps_reset_start_time = time.time()
        cur_state, norm_cur_state = env.reset_world(total_agentNum, full_observable_critic_flag, show=0)
        eps_reset_time_used = (time.time()-eps_reset_start_time)*1000
        # print("current episode {} reset time used is {} milliseconds".format(episode, eps_reset_time_used))  # need to + 1 here, or else will misrecord as the previous episode
        step_collision_record = [[] for _ in range(total_agentNum)]  # reset at each episode, so that we can record down collision at each step for each agent.
        episode_decision = [False] * 3
        agents_added = []
        eps_reward = []
        eps_noise = []
        step_time_breakdown = []
        single_eps_critic_cal_record = []
        
        cur_actor_hiddens = []
        for hidden_dim in actor_hidden_state_list:
            cur_actor_hiddens.append(np.zeros((hidden_dim)))

        # print("current episode is {}, scaling factor is {}".format(episode, model.var[0]))
        step = 0
        agent_added = 0  # this is an initialization for each new episode
        accum_reward = 0

        trajectory_eachPlay = []

        while True:  # start of a step
            if args.mode == "train":
                step_start_time = time.time()
                step_reward_record = [None] * n_agents

                noise_flag = True
                # noise_flag = False
                # populate gru history
                gru_history.append(np.array(norm_cur_state[0]))

                step_obtain_action_time_start = time.time()
                action, step_noise_val, cur_actor_hiddens, next_actor_hiddens = model.choose_action(norm_cur_state, total_step, episode, step, eps_end, noise_start_level, cur_actor_hiddens, use_allNeigh_wRadar, use_selfATT_with_radar, own_obs_only, noisy=noise_flag, use_GRU_flag=use_GRU_flag)  # noisy is false because we are using stochastic policy

                generate_action_time = (time.time() - step_obtain_action_time_start)*1000
                # print("current step obtain action time used is {} milliseconds".format(generate_action_time))

                # action = model.choose_action(cur_state, episode, noisy=True)

                one_step_transition_start = time.time()
                next_state, norm_next_state, polygons_list, all_agent_st_points, all_agent_ed_points, all_agent_intersection_point_list, all_agent_line_collection, all_agent_mini_intersection_list = env.step(action, step, acc_max, args, evaluation_by_episode, full_observable_critic_flag)
                step_transition_time = (time.time() - one_step_transition_start)*1000
                # print("current step transition time used is {} milliseconds".format(step_transition_time))

                # reward_aft_action, done_aft_action, check_goal, step_reward_record, agents_added = env.get_step_reward_5_v3(step, step_reward_record)   # remove reached agent here
                # reward_aft_action, done_aft_action, check_goal, step_reward_record = env.get_step_reward_5_v3(step, step_reward_record)   # remove reached agent here

                one_step_reward_start = time.time()
                # reward_aft_action, done_aft_action, check_goal, step_reward_record, status_holder, step_collision_record, bound_building_check = env.ss_reward_Mar(step, step_reward_record, step_collision_record, dummy_xy, full_observable_critic_flag, args, evaluation_by_episode)   # remove reached agent here
                reward_aft_action, done_aft_action, check_goal, step_reward_record, status_holder, step_collision_record, bound_building_check = env.ss_reward_2024(step, step_reward_record, step_collision_record, dummy_xy, full_observable_critic_flag, args, evaluation_by_episode)   # remove reached agent here
                reward_generation_time = (time.time() - one_step_reward_start)*1000
                # print("current step reward time used is {} milliseconds".format(reward_generation_time))

                step += 1  # current play step
                total_step += 1  # steps taken from 1st episode
                eps_noise.append(step_noise_val)
                traj_step_list = []
                for each_agent_idx, each_agent in env.all_agents.items():
                    traj_step_list.append([each_agent.pos[0], each_agent.pos[1], np.array(step_reward_record[each_agent_idx][1])])
                trajectory_eachPlay.append(traj_step_list)
                if len(gru_history) >= gru_history_length:
                    obs = []
                    next_obs = []
                    # ------------- to store norm or non-norm state into experience replay ---------------
                    for elementIdx, element in enumerate(norm_cur_state):
                    # for elementIdx, element in enumerate(cur_state):
                        if elementIdx != len(norm_cur_state)-1:  # meaning is not the last element
                        # if elementIdx != len(cur_state)-1:  # meaning is not the last element
                        #     obs.append(torch.from_numpy(np.stack(element)).data.float().to(device))
                            obs.append(torch.from_numpy(np.stack(element)).to(device))
                            # obs.append(np.stack(element))
                        else:
                            sur_agents = []
                            for each_agent_list in element:
                                # sur_agents.append(torch.from_numpy(np.squeeze(np.array(each_agent_list), axis=1)).float())
                                sur_agents.append(np.squeeze(np.array(each_agent_list), axis=1))
                            obs.append(sur_agents)

                    for elementIdx, element in enumerate(norm_next_state):
                    # for elementIdx, element in enumerate(cur_state):
                        if elementIdx != len(norm_next_state)-1:  # meaning is not the last element
                        # if elementIdx != len(cur_state)-1:  # meaning is not the last element
                        #     next_obs.append(torch.from_numpy(np.stack(element)).data.float().to(device))
                            next_obs.append(torch.from_numpy(np.stack(element)).to(device))
                            # next_obs.append(np.stack(element))
                        else:
                            sur_agents = []
                            for each_agent_list in element:
                                # sur_agents.append(torch.from_numpy(np.squeeze(np.array(each_agent_list), axis=1)).float())
                                sur_agents.append(torch.from_numpy(np.squeeze(np.array(each_agent_list), axis=1)))
                            next_obs.append(sur_agents)
                    # ------------------ end of store norm or non-norm state into experience replay --------------------
                    rw_tensor = torch.tensor(np.array(reward_aft_action), device=device)
                    # rw_tensor = np.array(reward_aft_action)
                    # rw_tensor = torch.FloatTensor(np.array(reward_aft_action)).to(device)
                    ac_tensor = torch.tensor(action, device=device)
                    # ac_tensor = action
                    # ac_tensor = torch.FloatTensor(action).to(device)
                    if full_observable_critic_flag:
                        eps_termination = 1.0 if any(done_aft_action) else 0.0
                        done_tensor = torch.tensor(np.array(eps_termination), device=device)
                    else:
                        done_aft_action = [int(value) for value in done_aft_action]
                        done_tensor = torch.tensor(np.array(done_aft_action), device=device)
                    # done_tensor = np.array(done_aft_action)
                    # done_tensor = torch.FloatTensor(done_aft_action).to(device)
                    # prepare hidden state information
                    # history_tensor = torch.FloatTensor(np.array(gru_history)).to(device)
                    # history_tensor = np.array(gru_history)
                    history_tensor = torch.tensor(np.array(gru_history), device=device)

                    # padded_tensor = torch.nn.functional.pad(hs_tensor, pad=(0, 0, 0, 0, 0, args.episode_length), mode='constant', value=0)
                    if full_observable_critic_flag:
                        # model.memory.push(obs, ac_tensor, next_obs, rw_tensor, done_tensor, history_tensor, cur_actor_hiddens, next_actor_hiddens)
                        model.memory.push(obs[0], obs[1], obs[2], ac_tensor, next_obs[0], next_obs[1], next_obs[2], rw_tensor, done_tensor, history_tensor, cur_actor_hiddens, next_actor_hiddens)
                    else:
                        # ------- push to memory one by one ----------
                        # for obs and next_obs
                        one_agent_obs = []
                        for i in range(total_agentNum):
                            one_agent_one_portion = []
                            for observation_portion in obs:
                                if isinstance(observation_portion, list):
                                    one_agent_one_portion.append(observation_portion[i])
                                else:
                                    one_agent_one_portion.append(observation_portion[i, :])
                            one_agent_obs.append(one_agent_one_portion)
                        one_agent_next_obs = []
                        for i in range(total_agentNum):
                            one_agent_one_portion = []
                            for observation_portion in next_obs:
                                if isinstance(observation_portion, list):
                                    one_agent_one_portion.append(observation_portion[i])
                                else:
                                    one_agent_one_portion.append(observation_portion[i, :])
                            one_agent_next_obs.append(one_agent_one_portion)

                        for i in range(len(one_agent_next_obs)):
                            # if done_tensor[i] == 1:
                            #     continue
                            # model.memory.push(one_agent_obs[i], ac_tensor[i, :], one_agent_next_obs[i], rw_tensor[i], done_tensor[i], history_tensor[:,i,:],
                            #                   cur_actor_hiddens[i, :], next_actor_hiddens[i,:])
                            model.memory.push(one_agent_obs[i][0], one_agent_obs[i][1], one_agent_obs[i][2], ac_tensor[i, :], one_agent_next_obs[i][0], one_agent_next_obs[i][1], one_agent_next_obs[i][2], rw_tensor[i], done_tensor[i], history_tensor[:,i,:],
                                              cur_actor_hiddens[i, :], next_actor_hiddens[i,:])
                        # ------- end of push to memory one by one ----------

                # accum_reward = accum_reward + reward_aft_action[0]  # we just take the first agent's reward, because we are using a joint reward, so all agents obtain the same reward.
                if full_observable_critic_flag:
                    accum_reward = accum_reward + reward_aft_action[0]  # when using combine critic, all 3 agent's reward are the same, we just need to record 1.
                else:
                    accum_reward = accum_reward + sum(reward_aft_action)

                step_update_time_start = time.time()
                c_loss, a_loss, single_eps_critic_cal_record = model.update_myown(episode, total_step, args.update_step, single_eps_critic_cal_record, transfer_learning, use_allNeigh_wRadar, use_selfATT_with_radar, wandb, full_observable_critic_flag, use_GRU_flag)  # last working learning framework
                update_time_used = (time.time() - step_update_time_start)*1000
                # print("current step {} update time used is {} milliseconds".format(step, update_time_used))
                cur_state = next_state
                norm_cur_state = norm_next_state
                cur_actor_hiddens = next_actor_hiddens
                eps_reward.append(step_reward_record)
                whole_step_time = (time.time()-step_start_time)*1000
                # print("current episode, one whole step time used is {} milliseconds".format(whole_step_time))
                step_time_breakdown.append([generate_action_time, step_transition_time, reward_generation_time,
                                            update_time_used, whole_step_time])
                if args.episode_length < step:
                    episode_decision[0] = True
                    print("Agents stuck in some places, maximum step in one episode reached, current episode {} ends, all {} steps used".format(episode, args.episode_length))
                elif (True in done_aft_action):
                    episode_decision[1] = True
                    print("Some agent triggers termination condition like collision, current episode {} ends at step {}".format(episode, step-1))  # we need to -1 here, because we perform step + 1 after each complete step. Just to be consistent with the step count inside the reward function.
                # elif all([agent.reach_target for agent_idx, agent in env.all_agents.items()]):
                elif all(check_goal):
                    episode_decision[2] = True
                    print("All agents have reached their destinations at step {}, episode {} terminated.".format(step-1, episode))
                elif all([agent.reach_target for agent_idx, agent in env.all_agents.items()]):  # check whether these two termination condition has any difference
                    episode_decision[2] = True
                    print(
                        "All agents have reached their destinations at step {}, episode {} terminated.".format(step - 1,
                                                                                                               episode))
                if True in episode_decision:

                    # end of an episode starts here

                    # time_used = time.time() - start_time
                    # print("update function used {} seconds to run".format(time_used))
                    # here onwards is end of an episode's play
                    score_history.append(accum_reward)

                    # print("[Episode %05d] reward %6.4f time used is %.2f sec" % (episode, accum_reward, time_used))
                    print("[Episode %05d] reward %6.4f" % (episode, accum_reward))

                    if use_wanDB:
                        wandb.log({'overall_reward': float(accum_reward)}, step=episode)
                        if c_loss and a_loss:
                            for idx, val in enumerate(c_loss):
                                # print(" agent %s, a_loss %3.2f c_loss %3.2f" % (idx, a_loss[idx].item(), c_loss[idx].item()))
                                wandb.log({'agent' + str(idx) + 'actor_loss': float(a_loss[idx].item()),
                                           'agent' + str(idx) + 'critic_loss': float(c_loss[idx].item())}, step=episode)
                    if True in done_aft_action and step < args.episode_length:
                        collision_count = collision_count + 1
                        if bound_building_check[0] == True:  # collide due to boundary
                            crash_to_bound = crash_to_bound + 1
                        elif bound_building_check[1] == True:  # collide due to building
                            crash_to_building = crash_to_building + 1
                        elif bound_building_check[2] == True:  # collide due to drones
                            crash_to_drone = crash_to_drone + 1
                            if bound_building_check[3] == True:
                                crash_due_to_nearest = crash_due_to_nearest + 1
                        else:
                            pass
                    else:  # no collision -> no True in done_aft_action, and all steps used
                        all_steps_used = all_steps_used + 1

                    if True in episode_goal_found:
                        # Count the number of reach cases
                        num_true = sum(episode_goal_found)
                        # Determine the number of True values and print the appropriate response
                        if num_true == 1:
                            # print("There is one True value in the list.")
                            one_drone_reach = one_drone_reach + 1
                        elif num_true == 2:
                            # print("There are two True values in the list.")
                            two_drone_reach = two_drone_reach + 1
                        else:  # all 3 reaches goal
                            all_drone_reach = all_drone_reach + 1
                            # print("There are no True values in the list.")

                    if episode % 100 == 0:  # every 100 episode we record the training performance (without evaluation)
                        # if episode == 10:
                        # After the loop, save the file once
                        writer._save()
                        print(f'Data saved to {excel_file_path}')
                        # save a gif every 100 episode during training
                        episode_to_check = str(episode)
                        save_gif(env, trajectory_eachPlay, plot_file_name, episode_to_check, episode)
                        print("collision count for last 100 episode is {}, {}%".format(collision_count,
                                                                        round(collision_count / 100 * 100,
                                                                              2)))
                        print("Collision due to bound is {}".format(crash_to_bound))
                        print("Collision due to building is {}".format(crash_to_building))
                        print("Collision due to drone is {}, among them, caused by nearest drone is {}".format(
                            crash_to_drone, crash_due_to_nearest))
                        print("all steps used count is {}, {}%".format(all_steps_used,
                                                                       round(all_steps_used / 100 * 100,
                                                                             2)))
                        print("One goal reached count is {}, {}%".format(one_drone_reach, round(
                            one_drone_reach / args.max_episodes * 100, 2)))
                        print("Two goal reached count is {}, {}%".format(two_drone_reach, round(
                            two_drone_reach / args.max_episodes * 100, 2)))
                        print("All goal reached count is {}, {}%".format(all_drone_reach, round(
                            all_drone_reach / args.max_episodes * 100, 2)))

                        collision_count = 0
                        one_drone_reach = 0
                        two_drone_reach = 0
                        three_drone_reach = 0
                        four_drone_reach = 0
                        five_drone_reach = 0
                        six_drone_reach = 0
                        seven_drone_reach = 0
                        all_drone_reach = 0
                        all_steps_used = 0
                        crash_to_bound = 0
                        crash_to_building = 0
                        crash_to_drone = 0
                        crash_due_to_nearest = 0

                    if episode % args.save_interval == 0 and args.mode == "train":
                    # if episode % 1 == 0 and args.mode == "train":
                        save_model = time.time()
                        # save the models at a predefined interval
                        # save model to my own directory
                        filepath = file_name+'/interval_record_eps'
                        model.save_model(episode, filepath)  # this is the original save model
                        time_used_for_csv_model_save = (time.time()-save_model)*1000  # *1000 for milliseconds
                        print("current episode used time in save csv and model is {} milliseconds".format(episode, time_used_for_csv_model_save))
                    # save episodes reward for entire system at each of one episode
                    eps_reward_record.append(eps_reward)
                    eps_check_collision.append(step_collision_record)
                    eps_noise_record.append(eps_noise)
                    episode_critic_loss_cal_record.append(single_eps_critic_cal_record)
                    # with open(r'episode_critic_loss_cal_record_1500.pickle', 'wb') as handle:
                    #     pickle.dump(episode_critic_loss_cal_record, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    epsTime = time.time()-episode_start_time
                    eps_time_record.append([eps_reset_time_used, epsTime, step_time_breakdown])
                    # print("episode {} used time in calculation is  {} seconds".format(episode, epsTime))

                    break  # this is to break out from "while True:", which is one play
            elif args.mode == "eval":

                step_reward_record = [None] * n_agents
                # show_step_by_step = True
                show_step_by_step = False
                saved_gif = True  # Don't save gif while doing mass run
                # saved_gif = False
                noise_flag = False
                # populate gru history
                gru_history.append(np.array(norm_cur_state[0]))

                # action, step_noise_val = model.choose_action(norm_cur_state, total_step, episode, step, eps_end, noise_start_level, gru_history, noisy=False) # noisy is false because we are using stochastic policy
                action, step_noise_val, cur_actor_hiddens, \
                next_actor_hiddens = model.choose_action(norm_cur_state, total_step, episode, step, eps_end, noise_start_level, cur_actor_hiddens, use_allNeigh_wRadar, use_selfATT_with_radar, own_obs_only, noisy=noise_flag, use_GRU_flag=use_GRU_flag)  # noisy is false because we are using stochastic policy

                # nearest_two_drones
                next_state, norm_next_state, polygons_list, all_agent_st_points, all_agent_ed_points, all_agent_intersection_point_list, all_agent_line_collection, all_agent_mini_intersection_list = env.step(action, step, acc_max, args, evaluation_by_episode, full_observable_critic_flag)  # no heading update here
                # reward_aft_action, done_aft_action, check_goal, step_reward_record, eps_status_holder, step_collision_record, bound_building_check = env.ss_reward(step, step_reward_record, step_collision_record, dummy_xy, full_observable_critic_flag, args, evaluation_by_episode, own_obs_only)
                reward_aft_action, done_aft_action, check_goal, step_reward_record, eps_status_holder, step_collision_record, bound_building_check = env.ss_reward_Mar(step, step_reward_record, step_collision_record, dummy_xy, full_observable_critic_flag, args, evaluation_by_episode)
                # reward_aft_action, done_aft_action, check_goal, step_reward_record = env.get_step_reward_5_v3(step, step_reward_record)

                step += 1
                total_step += 1
                cur_state = next_state
                norm_cur_state = norm_next_state
                # trajectory_eachPlay.append([[each_agent_traj[0], each_agent_traj[1], reward_aft_action[each_agent_idx]] for each_agent_idx, each_agent_traj in enumerate(cur_state[0])])
                traj_step_list = []
                for each_agent_idx, each_agent in env.all_agents.items():
                    # traj_step_list.append([each_agent.pos[0], each_agent.pos[1], reward_aft_action[each_agent_idx]])
                    traj_step_list.append([each_agent.pos[0], each_agent.pos[1], np.array(step_reward_record[each_agent_idx][1]), eps_status_holder[each_agent_idx]])
                trajectory_eachPlay.append(traj_step_list)
                accum_reward = accum_reward + sum(reward_aft_action)
                # show states in text
                for agentIdx, agent in env.all_agents.items():
                    print("drone {}, next WP is {}, deviation from ref line is {}, ref_line_reward is {}, "
                          "actual dist to goal is {}, dist_goal_reward is {}, velocity is {}, step {} reward is {}"
                          .format(agentIdx, agent.goal[-1], eps_status_holder[agentIdx]['deviation_to_ref_line'],
                                  eps_status_holder[agentIdx]['deviation_to_ref_line_reward'], eps_status_holder[agentIdx]['Euclidean_dist_to_goal'],
                                  eps_status_holder[agentIdx]['goal_leading_reward'], eps_status_holder[agentIdx]['current_drone_speed'], step,
                                  reward_aft_action[agentIdx]))

                if show_step_by_step:
                    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
                    matplotlib.use('TkAgg')
                    fig, ax = plt.subplots(1, 1)
                    for agentIdx, agent in env.all_agents.items():
                        plt.plot(agent.pos[0], agent.pos[1], marker=MarkerStyle(">", fillstyle="right",
                                                                                transform=Affine2D().rotate_deg(
                                                                                    math.degrees(agent.heading))),
                                 color='y')
                        plt.text(agent.pos[0], agent.pos[1], agent.agent_name)
                        plt.text(agent.pos[0]+5, agent.pos[1], str(reward_aft_action[agentIdx]))
                        plt.text(agent.pos[0]+5, agent.pos[1]-1, str(eps_status_holder[agentIdx][-1][1]))
                        plt.text(agent.pos[0]+5, agent.pos[1]-2, str(eps_status_holder[agentIdx][-1][2]))
                        # plot self_circle of the drone
                        self_circle = Point(agent.pos[0], agent.pos[1]).buffer(agent.protectiveBound, cap_style='round')
                        grid_mat_Scir = shapelypoly_to_matpoly(self_circle, False, 'k')
                        ax.add_patch(grid_mat_Scir)

                        # plot drone's detection range
                        detec_circle = Point(agent.pos[0], agent.pos[1]).buffer(agent.detectionRange / 2, cap_style='round')
                        detec_circle_mat = shapelypoly_to_matpoly(detec_circle, False, 'r')
                        # ax.add_patch(detec_circle_mat)

                        # Plot each start point
                        for point_deg, point_pos in all_agent_st_points[agentIdx].items():
                            ax.plot(point_pos.x, point_pos.y, 'o', color='blue')

                        # Plot each end point
                        for point_deg, point_pos in all_agent_ed_points[agentIdx].items():
                            ax.plot(point_pos.x, point_pos.y, 'o', color='green')

                        # Plot the lines of the LineString
                        for lines in all_agent_line_collection[agentIdx]:
                            x, y = lines.xy
                            ax.plot(x, y, color='k', linewidth=2, solid_capstyle='round', zorder=2)

                        # point_counter = 0
                        # # Plot each intersection point
                        # for point in intersection_point_list:
                        #     for ea_pt in point.geoms:
                        #         point_counter = point_counter + 1
                        #         ax.plot(ea_pt.x, ea_pt.y, 'o', color='red')

                        # plot minimum intersection point
                        # for pt_dist, pt_pos in mini_intersection_list.items():
                        for pt_pos in all_agent_mini_intersection_list[agentIdx]:
                            if pt_pos.type == 'MultiPoint':
                                for ea_pt in pt_pos.geoms:
                                    ax.plot(ea_pt.x, ea_pt.y, 'o', color='yellow')
                            else:
                                ax.plot(pt_pos.x, pt_pos.y, 'o', color='red')

                                # min_dist = np.min(agent.observableSpace)
                                # near_building_penalty_coef = 3
                                # # turningPtConst = 12.5
                                # turningPtConst = 5
                                # if turningPtConst == 12.5:
                                #     c = 1.25
                                # elif turningPtConst == 5:
                                #     c = 2
                                # m = (0 - 1) / (
                                #             turningPtConst - agent.protectiveBound)  # we must consider drone's circle, because when min_distance is less than drone's radius, it is consider collision.
                                # if min_dist >= agent.protectiveBound and min_dist <= turningPtConst:
                                #     near_building_penalty = near_building_penalty_coef * (
                                #                 m * min_dist + c)  # at each step, penalty from 3 to 0.
                                # else:
                                #     near_building_penalty = 0  # if min_dist is outside of the bound, other parts of the reward will be taking care.
                                # if min_dist < agent.protectiveBound:
                                #     print("check for collision")
                                plt.text(pt_pos.x, pt_pos.y, eps_status_holder[agentIdx][-1][3], fontsize=12)

                        ini = agent.pos
                        for wp in agent.ref_line.coords:
                            plt.plot([wp[0], ini[0]], [wp[1], ini[1]], '--', color='c')
                            ini = wp

                    # draw occupied_poly
                    for one_poly in env.world_map_2D_polyList[0][0]:
                        one_poly_mat = shapelypoly_to_matpoly(one_poly, True, 'y', 'b')
                        ax.add_patch(one_poly_mat)
                    # draw non-occupied_poly
                    for zero_poly in env.world_map_2D_polyList[0][1]:
                        zero_poly_mat = shapelypoly_to_matpoly(zero_poly, False, 'y')
                        ax.add_patch(zero_poly_mat)

                    # show building obstacles
                    for poly in env.buildingPolygons:
                        matp_poly = shapelypoly_to_matpoly(poly, False, 'red')  # the 3rd parameter is the edge color
                        # ax.add_patch(matp_poly)

                    # # show the nearest building obstacles
                    # nearest_buildingPoly_mat = shapelypoly_to_matpoly(nearest_buildingPoly, True, 'g', 'k')
                    # ax.add_patch(nearest_buildingPoly_mat)

                    # plt.axvline(x=self.bound[0], c="green")
                    # plt.axvline(x=self.bound[1], c="green")
                    # plt.axhline(y=self.bound[2], c="green")
                    # plt.axhline(y=self.bound[3], c="green")

                    plt.xlabel("X axis")
                    plt.ylabel("Y axis")
                    plt.axis('equal')
                    plt.show()

                if args.episode_length < step or (True in done_aft_action) or all([agent.reach_target for agent_idx, agent in env.all_agents.items()]):  # when termination condition reached
                    # check if in this episode there are situation where agents found their goal
                    for agent_idx, agent in env.all_agents.items():
                        episode_goal_found[agent_idx] = agent.reach_target
                    # episode_goal_found = [for agents in env.all_agents]
                # if args.episode_length < step:  # when termination condition reached, without counting drone collision to buildings/wall
                    # display current episode out status through status_holder
                    for each_agent_idx, each_agent in enumerate(eps_status_holder):
                        for step_idx, step_reward_decomposition in enumerate(each_agent):
                            pass
                            # print(r"agent {}, step {}, distance to goal is {} m, goal reward is {}, ref line reward is {}, current step reward is {}.".format(each_agent_idx, step_idx, step_reward_decomposition[0], step_reward_decomposition[1], step_reward_decomposition[2], step_reward_decomposition[3]))
                            # print("near goal reward is {}".format(step_reward_decomposition[6]))
                            # print("current spd is {} m/s, curent spd penalty is {}". format(step_reward_decomposition[5], step_reward_decomposition[4]))
                    print("[Episode %05d] reward %6.4f " % (episode, accum_reward))

                    if get_evaluation_status:
                        if simply_view_evaluation:
                        # ------------------ static display trajectory ---------------------------- #
                            view_static_traj(env, trajectory_eachPlay)
                        # ------------------ end of static display trajectory ---------------------------- #

                        # ---------- new save as gif ----------------------- #
                        else:
                            save_gif(env, trajectory_eachPlay, pre_fix, episode_to_check, episode)
                    if evaluation_by_episode:
                        if True in done_aft_action and step < args.episode_length:
                            # save_gif(env, trajectory_eachPlay, pre_fix, episode_to_check, episode)
                            if saved_gif == False:
                                save_gif(env, trajectory_eachPlay, pre_fix, episode_to_check, episode)
                                saved_gif = True  # once current episode saved, no need to save one more time.
                            collision_count = collision_count + 1
                            # print("Episode {}, {} steps before collision".format(episode, step))
                            steps_before_collide.append(step)
                            if bound_building_check[0] == True:  # collide due to boundary
                                crash_to_bound = crash_to_bound + 1
                            elif bound_building_check[1] == True:  # collide due to building
                                crash_to_building = crash_to_building + 1
                            elif bound_building_check[2] == True:  # collide due to drones
                                crash_to_drone = crash_to_drone + 1
                                # save_gif(env, trajectory_eachPlay, pre_fix, episode_to_check, episode)
                                if bound_building_check[3] == True:
                                    crash_due_to_nearest = crash_due_to_nearest + 1
                            else:
                                pass

                        else:  # no collision -> no True in done_aft_action, and all steps used
                            # for each_agent in env.all_agents.values():
                            #     if each_agent.bound_collision == True:
                            #         collision_count = collision_count + 1
                            #         crash_to_bound = crash_to_bound + 1
                            #     elif each_agent.building_collision == True:
                            #         collision_count = collision_count + 1
                            #         crash_to_building = crash_to_building + 1
                            #     elif each_agent.drone_collision == True:
                            #         collision_count = collision_count + 1
                            #         crash_to_drone = crash_to_drone + 1
                            #     else:
                            #         pass
                            all_steps_used = all_steps_used + 1

                        if True in episode_goal_found:
                            # Count the number of reach cases
                            num_true = sum(episode_goal_found)
                            # Determine the number of True values and print the appropriate response
                            if num_true == 1:
                                if saved_gif == False:
                                    save_gif(env, trajectory_eachPlay, pre_fix, episode_to_check, episode)
                                    saved_gif = True  # once current episode saved, no need to save one more time.
                                # print("There is one True value in the list.")
                                one_drone_reach = one_drone_reach + 1
                            elif num_true == 2:
                                if saved_gif == False:
                                    save_gif(env, trajectory_eachPlay, pre_fix, episode_to_check, episode)
                                    saved_gif = True  # once current episode saved, no need to save one more time.
                                # print("There are two True values in the list.")
                                two_drone_reach = two_drone_reach + 1
                            elif num_true == 3:
                                three_drone_reach = three_drone_reach + 1
                            elif num_true == 4:
                                four_drone_reach = four_drone_reach + 1
                            elif num_true == 5:
                                five_drone_reach = five_drone_reach + 1
                            elif num_true == 6:
                                six_drone_reach = six_drone_reach + 1
                            elif num_true == 7:
                                seven_drone_reach = seven_drone_reach + 1
                            else:  # all 3 reaches goal
                                all_drone_reach = all_drone_reach + 1
                                # print("There are no True values in the list.")
                    else:  # evaluation by sorties, for each episode loop over all agent's status in current episode
                        for each_agent in env.all_agents.values():
                            if each_agent.bound_collision == True:
                                collision_count = collision_count + 1
                                crash_to_bound = crash_to_bound + 1
                            elif each_agent.building_collision == True:
                                collision_count = collision_count + 1
                                crash_to_building = crash_to_building + 1
                            elif each_agent.drone_collision == True:
                                collision_count = collision_count + 1
                                crash_to_drone = crash_to_drone + 1
                            elif each_agent.reach_target == True:
                                sorties_reached = sorties_reached + 1
                            else:
                                idle_drone = idle_drone + 1
                    break

    if args.mode == "train":  # only save pickle at end of training to save computational time.
        with open(plot_file_name + '/all_episode_reward.pickle', 'wb') as handle:
            pickle.dump(eps_reward_record, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(plot_file_name + '/all_episode_noise.pickle', 'wb') as handle:
            pickle.dump(eps_noise_record, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(plot_file_name + '/all_episode_time.pickle', 'wb') as handle:
            pickle.dump(eps_time_record, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(plot_file_name + '/all_episode_collision.pickle', 'wb') as handle:
            pickle.dump(eps_check_collision, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(file_name + '/GFG.csv', 'w') as f:
            # using csv.writer method from CSV package
            write = csv.writer(f)
            write.writerows([score_history])
        # After the loop, save the file once
        writer._save()
        print(f'Data saved to {excel_file_path}')
    else:
        if evaluation_by_episode:
            print("total collision count is {}, {}%".format(collision_count, round(collision_count/args.max_episodes*100,2)))
            print("Collision due to bound is {}".format(crash_to_bound))
            print("Collision due to building is {}".format(crash_to_building))
            # print("Collision due to drone is {}, among them, caused by nearest drone is {}".format(crash_to_drone, crash_due_to_nearest))
            print("Collision due to drone is {}, among them, caused by any of previous two nearest drone is {}".format(crash_to_drone, crash_due_to_nearest))
            print("all steps used count is {}, {}%".format(all_steps_used, round(all_steps_used/100*100, 2)))
            print("One goal reached count is {}, {}%".format(one_drone_reach, round(one_drone_reach/100*100, 2)))
            print("Two goal reached count is {}, {}%".format(two_drone_reach, round(two_drone_reach/100*100, 2)))
            print("Three goal reached count is {}, {}%".format(three_drone_reach, round(three_drone_reach/100*100, 2)))
            print("Four goal reached count is {}, {}%".format(four_drone_reach, round(four_drone_reach/100*100, 2)))
            print("Five goal reached count is {}, {}%".format(five_drone_reach, round(five_drone_reach/100*100, 2)))
            print("Six goal reached count is {}, {}%".format(six_drone_reach, round(six_drone_reach/100*100, 2)))
            print("Seven goal reached count is {}, {}%".format(seven_drone_reach, round(seven_drone_reach/100*100, 2)))
            print("All goal reached count is {}, {}%".format(all_drone_reach, round(all_drone_reach/100*100, 2)))
        else:
            print("Total collision {}".format(collision_count))
            print("Collision to bound {}".format(crash_to_bound))
            print("Collision to building {}".format(crash_to_building))
            print("Collision to drone {}".format(crash_to_drone))
            print("Destination reached {}".format(sorties_reached))
            print("Idle UAV {}".format(idle_drone))
    print(f'training finishes, time spent: {datetime.timedelta(seconds=int(time.time() - training_start_time))}')
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    matplotlib.use('TkAgg')
    # plot2 = plt.plot(steps_before_collide)
    plot2 = plt.scatter(range(len(steps_before_collide)), steps_before_collide)
    plt.grid(linestyle='-.')
    plt.xlabel('episodes')
    plt.ylabel('steps taken')
    plt.show()
    if use_wanDB:
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', default="simple_spread", type=str)
    parser.add_argument('--max_episodes', default=20000, type=int)  # run for a total of 50000 episodes
    parser.add_argument('--algo', default="maddpg", type=str, help="commnet/bicnet/maddpg")
    parser.add_argument('--mode', default="eval", type=str, help="train/eval")
    # parser.add_argument('--episode_length', default=150, type=int)  # maximum play per episode
    parser.add_argument('--episode_length', default=100, type=int)  # maximum play per episode
    # parser.add_argument('--episode_length', default=100, type=int)  # maximum play per episode
    parser.add_argument('--memory_length', default=int(1e5), type=int)
    # parser.add_argument('--memory_length', default=int(1e4), type=int)
    parser.add_argument('--seed', default=777, type=int)  # may choose to use 3407
    # parser.add_argument('--batch_size', default=10, type=int)  # original 512
    parser.add_argument('--batch_size', default=512, type=int)  # original 512
    # parser.add_argument('--batch_size', default=3, type=int)  # original 512
    # parser.add_argument('--batch_size', default=1536, type=int)  # original 512
    parser.add_argument('--gamma', default=0.95, type=float)
    parser.add_argument('--tau', default=0.01, type=float)
    parser.add_argument('--update_step', default=1, type=int)
    parser.add_argument('--render_flag', default=False, type=bool)
    parser.add_argument('--ou_theta', default=0.15, type=float)
    parser.add_argument('--ou_mu', default=0.0, type=float)
    parser.add_argument('--ou_sigma', default=0.2, type=float)
    parser.add_argument('--epsilon_decay', default=10000, type=int)
    parser.add_argument('--tensorboard', default=True, action="store_true")
    parser.add_argument("--save_interval", default=1000, type=int)  # save model for every 5000 episodes
    parser.add_argument("--model_episode", default=60000, type=int)
    parser.add_argument('--gru_history_length', default=10, type=int)  # original 1000
    parser.add_argument('--log_dir', default=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))

    args = parser.parse_args()

    main(args)
