# from Nnetworks_MADDPGv3 import CriticNetwork_0724, ActorNetwork
from Nnetworks_randomOD_radar_sur_drones_N_Model_use_tdCPA_forV2 import *
from Utilities_own_randomOD_radar_sur_drones_N_Model_use_tdCPA_forV2 import *
import torch
from copy import deepcopy
from torch.optim import Adam
from memory_randomOD_radar_sur_drones_N_Model_use_tdCPA_forV2 import ReplayMemory, Experience
# from random_process_MADDPGv3_randomOD import OrnsteinUhlenbeckProcess
from torch.autograd import Variable
import os
import pandas as pd
import torch.nn as nn
import time
import numpy as np
import torch as T
# from utils_randomOD_radar_sur_drones_oneModel_use_tdCPA import device
import csv

# device = torch.device('cuda')

def soft_update(target, source, t):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(
            (1 - t) * target_param.data + t * source_param.data)


def hard_update(target, source):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(source_param.data)


class MADDPG:
    def __init__(self, actor_dim, critic_dim, dim_act, actor_hidden_state_size, gru_history_length, n_agents,
                 args, cr_lr, ac_lr, gamma, tau, full_observable_critic_flag, use_GRU_flag, use_single_portion_selfATT,
                 use_selfATT_with_radar, use_allNeigh_wRadar, own_obs_only, normalizer, device):
        self.args = args
        self.mode = args.mode
        self.actors = []
        self.critics = []
        self.device = device
        # original
        # self.actors = [Actor(dim_obs, dim_act) for _ in range(n_agents)]
        # self.critics = [Critic(n_agents, dim_obs, dim_act) for _ in range(n_agents)]

        # self.actors = [Stocha_actor(actor_dim, dim_act) for _ in range(n_agents)]  # use stochastic policy
        # self.actors = [ActorNetwork_TwoPortion(actor_dim, dim_act) for _ in range(n_agents)]  # use deterministic policy
        # only construct one-model
        if full_observable_critic_flag:
            # self.actors = ActorNetwork_allnei_wRadar(actor_dim, dim_act)
            self.actors = [ActorNetwork_allnei_wRadar(actor_dim, dim_act) for _ in range(n_agents)]
        else:
            if use_GRU_flag:
                self.actors = ActorNetwork_GRU_TwoPortion(actor_dim, dim_act, actor_hidden_state_size)
            elif use_single_portion_selfATT:
                self.actors = ActorNetwork_ATT(actor_dim, dim_act)
            elif use_allNeigh_wRadar:
                if own_obs_only:
                    self.actors = ActorNetwork_obs_only(actor_dim, dim_act)
                else:
                    self.actors = ActorNetwork_allnei_wRadar(actor_dim, dim_act)
            elif use_selfATT_with_radar:
                self.actors = ActorNetwork_ATT_wRadar(actor_dim, dim_act)
            else:
                self.actors = ActorNetwork_TwoPortion(actor_dim, dim_act)
        # end of only construct one-model
        # self.actors = [ActorNetwork_OnePortion(actor_dim, dim_act) for _ in range(n_agents)]  # use deterministic policy
        # self.actors = [GRUCELL_actor(actor_dim, dim_act, actor_hidden_state_size) for _ in range(n_agents)]  # use deterministic with GRU module policy
        # self.critics = [CriticNetwork_0724(critic_dim, n_agents, dim_act) for _ in range(n_agents)]
        # self.critics = [CriticNetwork(critic_dim, n_agents, dim_act) for _ in range(n_agents)]
        # self.critics = [CriticNetwork_wGru(critic_dim, n_agents, dim_act, gru_history_length) for _ in range(n_agents)]
        if full_observable_critic_flag:
            # self.critics = [critic_combine_TwoPortion(critic_dim, n_agents, dim_act, gru_history_length,
            #                                           actor_hidden_state_size) for _ in range(n_agents)]

            # one model centralized critic
            self.critics = critic_combine_TwoPortion_fullneiWradar(critic_dim, n_agents, dim_act, gru_history_length, actor_hidden_state_size, normalizer, device)
        else:
            # self.critics = [critic_single_TwoPortion(critic_dim, n_agents, dim_act, gru_history_length, actor_hidden_state_size) for _ in range(n_agents)]
            # only construct one-model
            if use_GRU_flag:
                self.critics = critic_single_GRU_TwoPortion(critic_dim, n_agents, dim_act, gru_history_length,
                                                        actor_hidden_state_size)
            elif use_single_portion_selfATT:
                self.critics = critic_single_TwoPortion(critic_dim, n_agents, dim_act, gru_history_length,
                                                        actor_hidden_state_size)
            elif use_allNeigh_wRadar:
                if own_obs_only:
                    self.critics = critic_single_obs_only(critic_dim, n_agents, dim_act, gru_history_length,
                                                        actor_hidden_state_size)
                else:
                    self.critics = critic_single_TwoPortion_wRadar(critic_dim, n_agents, dim_act, gru_history_length,
                                                        actor_hidden_state_size)
            elif use_selfATT_with_radar:
                self.critics = critic_single_TwoPortion_wRadar(critic_dim, n_agents, dim_act, gru_history_length,
                                                        actor_hidden_state_size)
            else:
                self.critics = critic_single_TwoPortion(critic_dim, n_agents, dim_act, gru_history_length, actor_hidden_state_size)
            # end of only construct one-model
        # self.critics = [critic_single_OnePortion(critic_dim, n_agents, dim_act, gru_history_length, actor_hidden_state_size) for _ in range(n_agents)]

        self.n_agents = n_agents
        self.n_actor_dim = actor_dim
        self.n_critic_dim = critic_dim
        self.n_actions = dim_act

        # self.actors_target = deepcopy(self.actors)
        self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)

        self.memory = ReplayMemory(args.memory_length)
        self.batch_size = args.batch_size
        self.use_cuda = torch.cuda.is_available()
        # self.episodes_before_train = args.episode_before_train

        # self.GAMMA = 0.95  # original
        # self.tau = 0.01  # original

        self.GAMMA = gamma
        self.tau = tau

        self.var = [1.0 for i in range(n_agents)]
        # self.OU_noise = OrnsteinUhlenbeckProcess(dim_act)
        self.OU_noise = OUActionNoise(np.zeros(dim_act), 0.1, 0.15, 0.5)
        # self.var = [0.5 for i in range(n_agents)]

        # original, critic learning rate is 10 times larger compared to actor
        # self.critic_optimizer = [Adam(x.parameters(), lr=0.001) for x in self.critics]
        # self.actor_optimizer = [Adam(x.parameters(), lr=0.0001) for x in self.actors]

        # self.critic_optimizer = [Adam(x.parameters(), lr=cr_lr) for x in self.critics]
        # self.actor_optimizer = [Adam(x.parameters(), lr=ac_lr) for x in self.actors]

        # only construct one-model
        # self.critic_optimizer = Adam(self.critics.parameters(), lr=cr_lr)
        self.critic_optimizer = [Adam(x.parameters(), lr=cr_lr) for x in self.critics] if isinstance(self.critics, list) else Adam(self.critics.parameters(), lr=cr_lr)
        # self.actor_optimizer = Adam(self.actors.parameters(), lr=ac_lr)
        self.actor_optimizer = [Adam(x.parameters(), lr=ac_lr) for x in self.actors] if isinstance(self.actors, list) else Adam(self.actors.parameters(), lr=ac_lr)
        # end of only construct one-model

        if self.device.type == 'cuda':
            if isinstance(self.actors, list):
                for act_model_idx, act_model in enumerate(self.actors):
                    self.actors[act_model_idx].cuda().to(dtype=torch.float64)
                    self.actors_target[act_model_idx].cuda().to(dtype=torch.float64)
            else:
                self.actors.cuda().to(dtype=torch.float64)
                self.actors_target.cuda().to(dtype=torch.float64)

            if isinstance(self.critics, list):
                for critic_model_idx, critic_model in enumerate(self.critics):
                    self.critics[critic_model_idx].cuda().to(dtype=torch.float64)
                    self.critics_target[critic_model_idx].cuda().to(dtype=torch.float64)
            else:
                self.critics.cuda().to(dtype=torch.float64)
                self.critics_target.cuda().to(dtype=torch.float64)
        else:
            if isinstance(self.actors, list):
                for act_model_idx, act_model in enumerate(self.actors):
                    self.actors[act_model_idx].to(dtype=torch.float64)
                    self.actors_target[act_model_idx].to(dtype=torch.float64)
            else:
                self.actors.to(dtype=torch.float64)
                self.actors_target.to(dtype=torch.float64)

            if isinstance(self.critics, list):
                for critic_model_idx, critic_model in enumerate(self.critics):
                    self.critics[critic_model_idx].to(dtype=torch.float64)
                    self.critics_target[critic_model_idx].to(dtype=torch.float64)
            else:
                self.critics.to(dtype=torch.float64)
                self.critics_target.to(dtype=torch.float64)

        self.steps_done = 0
        self.episode_done = 0

    def load_model(self, filePath, full_observable_critic_flag):
        if self.args.model_episode:
            path_flag = True
            # for idx in range(self.n_agents):
            #     path_flag = path_flag \
            #                 and (os.path.exists("trained_model/maddpg/actor["+ str(idx) + "]_"
            #                                     +str(self.args.model_episode)+".pth")) \
            #                 and (os.path.exists("trained_model/maddpg/critic["+ str(idx) + "]_"
            #                                     +str(self.args.model_episode)+".pth"))

            if path_flag:
                print("load model!")
                # for idx in range(self.n_agents):
                #     actor = torch.load("trained_model/maddpg/actor["+ str(idx) + "]_"+str(self.args.model_episode)+".pth")
                #     critic = torch.load("trained_model/maddpg/critic["+ str(idx) + "]_"+str(self.args.model_episode)+".pth")
                #     self.actors[idx].load_state_dict(actor.state_dict())
                #     self.critics[idx].load_state_dict(critic.state_dict())
                for path_idx, path in enumerate(filePath):
                    if full_observable_critic_flag:
                        self.actors[path_idx].load_state_dict(torch.load(path))
                    else:
                        self.actors.load_state_dict(torch.load(path))



        self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)

    def save_model(self, episode, file_path):
        # if not os.path.exists("./trained_model_myenv/"):
        #     os.mkdir("./trained_model_myenv/")
        # if not os.path.exists("./trained_model/" + str(self.args.algo) + "/"):
        #     # os.mkdir(r"F:\githubClone\MAProj_myversion\algo/trained_model/" + str(self.args.algo))
        #     os.mkdir(r"D:\Multi_agent_AAC\old_framework_test\algo/trained_model/" + str(self.args.algo))
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        if isinstance(self.actors, list):
            for actor in self.actors:
                torch.save(actor.state_dict(),
                           file_path + '/' + 'episode_' + str(episode) + '_' + 'actor_net.pth')
        else:
            torch.save(self.actors.state_dict(), file_path + '/' +'episode_'+str(episode)+'_' + 'actor_net.pth')

    def update_myown(self, i_episode, total_step_count, UPDATE_EVERY, single_eps_critic_cal_record, transfer_learning, use_allNeigh_wRadar, use_selfATT_with_radar, wandb=None, full_observable_critic_flag=False, use_GRU_flag=False):

        self.train_num = i_episode

        if len(self.memory) <= self.batch_size:
        # if len(self.memory) < self.memory.capacity:
        # if True:
            return None, None, single_eps_critic_cal_record
        # print("learning starts")
        # if i_episode > 13000:
        #     changed_lr = 0.0005
        #     for param_group in self.critic_optimizer.param_groups:
        #         param_group['lr'] = changed_lr
        #     for param_group in self.actor_optimizer.param_groups:
        #         param_group['lr'] = changed_lr

        if self.device.type == 'cuda':
            BoolTensor = torch.cuda.BoolTensor
            FloatTensor = torch.cuda.FloatTensor
        else:
            BoolTensor = torch.BoolTensor
            FloatTensor = torch.FloatTensor

        c_loss = []
        a_loss = []

        if full_observable_critic_flag:
            var_iteration = self.n_agents
        else:
            var_iteration = 1

        for agent in range(var_iteration):
            if full_observable_critic_flag and isinstance(self.actors, list):
                if agent == 0:
                    transitions = self.memory.sample(self.batch_size)
                    batch = Experience(*zip(*transitions))

                    action_batch = torch.stack(batch.actions)

                    stacked_elem_0_combine = torch.stack(batch.states_obs)
                    stacked_elem_1_combine = torch.stack(batch.states_nei)
                    stacked_elem_2_combine = torch.stack(batch.states_grid)

                    next_stacked_elem_0_combine = torch.stack(batch.next_states_obs)
                    next_stacked_elem_1_combine = torch.stack(batch.next_states_nei)
                    next_stacked_elem_2_combine = torch.stack(batch.next_states_grid)

                    reward_batch = torch.stack(batch.rewards)
                    done_combined = torch.stack(batch.dones)
                    non_final_next_states_actorin = [next_stacked_elem_0_combine, next_stacked_elem_1_combine, next_stacked_elem_2_combine]
            else:
                transitions = self.memory.sample(self.batch_size)
                batch = Experience(*zip(*transitions))

                # stack tensors only once
                stacked_elem_0 = torch.stack(batch.states_obs)
                stacked_elem_1 = torch.stack(batch.states_nei)
                stacked_elem_2 = torch.stack(batch.states_grid)

                # for next state
                next_stacked_elem_0 = torch.stack(batch.next_states_obs)
                next_stacked_elem_1 = torch.stack(batch.next_states_nei)
                next_stacked_elem_2 = torch.stack(batch.next_states_grid)

                dones_stacked = torch.stack(batch.dones)
                reward_batch = torch.stack(batch.rewards)
                action_batch = torch.stack(batch.actions)
                non_final_next_states_actorin = [next_stacked_elem_0, next_stacked_elem_1, next_stacked_elem_2]


            if use_GRU_flag:
                agents_next_hidden_state = torch.stack(batch.next_hidden).type(FloatTensor)
                agents_cur_hidden_state = torch.stack(batch.cur_hidden).type(FloatTensor)
            # whole_ownState = stacked_elem_0_combine  # own_state only

            # non_final_next_states_actorin = [next_stacked_elem_0]  # 2 portion available
            # non_final_next_states_actorin = [next_stacked_elem_0, next_stacked_elem_1]  # 2 portion available


            # configured for target Q

            # whole_curren_action = action_batch.view(self.batch_size, -1)
            whole_curren_action = action_batch

            # non_final_next_actions = [self.actors_target[i](non_final_next_states_actorin[0][:,i,:], history_batch[:,:,i,:])[0] for i in range(self.n_agents)]
            # non_final_next_actions = [self.actors_target[i](non_final_next_states_actorin[0][:,i,:], agents_next_hidden_state[:,i,:])[0] for i in range(self.n_agents)]
            # using one model #
            # non_final_next_actions = [self.actors_target([non_final_next_states_actorin[0][:,i,:],
            #                                               non_final_next_states_actorin[1][:,i,:]]) for i in range(self.n_agents)]
            if full_observable_critic_flag:
                non_final_next_actions = [self.actors_target[i](
                    [non_final_next_states_actorin[0][:, i, :], non_final_next_states_actorin[1][:, i, :],
                     non_final_next_states_actorin[2][:, i, :]]) for i in
                 range(self.n_agents)]
                non_final_next_actions = torch.stack(non_final_next_actions, dim=1)
            else:
                if use_GRU_flag:
                    non_final_next_actions = self.actors_target(
                        [non_final_next_states_actorin[0], non_final_next_states_actorin[1]], agents_next_hidden_state)[0]
                elif use_selfATT_with_radar or use_allNeigh_wRadar:
                    non_final_next_actions = self.actors_target([non_final_next_states_actorin[0],
                                                                 non_final_next_states_actorin[1],
                                                                 non_final_next_states_actorin[2]])
                else:
                    non_final_next_actions = self.actors_target([non_final_next_states_actorin[0], non_final_next_states_actorin[1]])

            # non_final_next_combine_actions = torch.stack(non_final_next_actions).view(self.batch_size, -1)
            non_final_next_combine_actions = non_final_next_actions

            # get current Q-estimate, using agent's critic network
            # current_Q = self.critics[agent](whole_state, whole_action, whole_agent_combine_gru)
            # current_Q = self.critics[agent](whole_state, whole_action, history_batch[:, :, agent, :])
            if full_observable_critic_flag:
                current_Q = self.critics([stacked_elem_0_combine, stacked_elem_1_combine, stacked_elem_2_combine], whole_curren_action)
            else:
                # current_Q = self.critics[agent]([stacked_elem_0[:,agent,:], stacked_elem_1[:,agent,:]], action_batch[:,agent,:])
                # using one model
                if use_GRU_flag:
                    current_Q = self.critics([stacked_elem_0, stacked_elem_1], action_batch, agents_cur_hidden_state)[0]
                elif use_selfATT_with_radar or use_allNeigh_wRadar:
                    current_Q = self.critics([stacked_elem_0, stacked_elem_1, stacked_elem_2], action_batch)
                else:
                    current_Q = self.critics([stacked_elem_0, stacked_elem_1], action_batch)

            # has_positive_values = (current_Q > 0).any()
            # if has_positive_values:
            #     print("true")
            with T.no_grad():
                # next_target_critic_value = self.critics_target[agent](next_stacked_elem_0_combine, non_final_next_actions.view(-1,self.n_agents * self.n_actions), whole_agent_combine_gru).squeeze()
                # next_target_critic_value = self.critics_target[agent](next_stacked_elem_0_combine, non_final_next_actions.view(-1,self.n_agents * self.n_actions), history_batch[:, :, agent, :]).squeeze()
                if full_observable_critic_flag:
                    next_target_critic_value = self.critics_target(
                        [next_stacked_elem_0_combine, next_stacked_elem_1_combine, next_stacked_elem_2_combine],
                        non_final_next_combine_actions).squeeze()
                else:
                    # next_target_critic_value = self.critics_target[agent](
                    #     [next_stacked_elem_0[:, agent, :], next_stacked_elem_1[:, agent, :]],
                    #     non_final_next_actions[agent]).squeeze()
                    # using one model
                    if use_GRU_flag:
                        next_target_critic_value = self.critics_target([next_stacked_elem_0, next_stacked_elem_1],
                            non_final_next_actions, agents_next_hidden_state)[0].squeeze()
                    elif use_selfATT_with_radar or use_allNeigh_wRadar:
                        next_target_critic_value = self.critics_target([next_stacked_elem_0, next_stacked_elem_1, next_stacked_elem_2],
                            non_final_next_actions).squeeze()
                    else:
                        next_target_critic_value = self.critics_target([next_stacked_elem_0, next_stacked_elem_1],
                            non_final_next_actions).squeeze()

                # tar_Q_before_rew = self.GAMMA * next_target_critic_value * (1-dones_stacked[:, agent])
                # reward_cal = reward_batch[:, agent].clone()
                reward_cal = reward_batch.clone()  # for one model
                if full_observable_critic_flag:
                    tar_Q_before_rew = self.GAMMA * next_target_critic_value * (1 - done_combined)  # for one model
                    target_Q = (reward_batch[:, agent]) + (
                                self.GAMMA * next_target_critic_value * (1 - done_combined))
                else:
                    tar_Q_before_rew = self.GAMMA * next_target_critic_value * (1 - dones_stacked)  # for one model
                    target_Q = (reward_batch) + (self.GAMMA * next_target_critic_value * (1-dones_stacked))
                target_Q = target_Q.unsqueeze(1)
                tar_Q_after_rew = target_Q.clone()
            if full_observable_critic_flag and agent==0:
                # critic_time = time.time()
                loss_Q = nn.MSELoss()(current_Q, target_Q.detach())
                cal_loss_Q = loss_Q.clone()
                single_eps_critic_cal_record.append([tar_Q_before_rew.detach().cpu().numpy(),
                                                     reward_cal.detach().cpu().numpy(),
                                                     tar_Q_after_rew.detach().cpu().numpy(),
                                                     cal_loss_Q.detach().cpu().numpy(),
                                                     (tar_Q_before_rew.detach().cpu().numpy().min(), tar_Q_before_rew.detach().cpu().numpy().max()),
                                                     (reward_cal.detach().cpu().numpy().min(), reward_cal.detach().cpu().numpy().max()),
                                                     (tar_Q_after_rew.detach().cpu().numpy().min(), tar_Q_after_rew.detach().cpu().numpy().max()),
                                                     (cal_loss_Q.detach().cpu().numpy().min(), cal_loss_Q.detach().cpu().numpy().max())])

                # self.critic_optimizer[agent].zero_grad()
                self.critic_optimizer.zero_grad()
                loss_Q.backward()
                # self.has_gradients(self.critics[agent], agent, wandb)
                # self.critic_optimizer[agent].step()
                self.critic_optimizer.step()
                # end_critic_update = (time.time() - critic_time)*1000
                # print("time used in critic update is {} ms".format(end_critic_update))
            else:
                loss_Q = nn.MSELoss()(current_Q, target_Q.detach())
                cal_loss_Q = loss_Q.clone()
                single_eps_critic_cal_record.append([tar_Q_before_rew.detach().cpu().numpy(),
                                                     reward_cal.detach().cpu().numpy(),
                                                     tar_Q_after_rew.detach().cpu().numpy(),
                                                     cal_loss_Q.detach().cpu().numpy(),
                                                     (tar_Q_before_rew.detach().cpu().numpy().min(), tar_Q_before_rew.detach().cpu().numpy().max()),
                                                     (reward_cal.detach().cpu().numpy().min(), reward_cal.detach().cpu().numpy().max()),
                                                     (tar_Q_after_rew.detach().cpu().numpy().min(), tar_Q_after_rew.detach().cpu().numpy().max()),
                                                     (cal_loss_Q.detach().cpu().numpy().min(), cal_loss_Q.detach().cpu().numpy().max())])
                self.critic_optimizer.zero_grad()
                loss_Q.backward()
                self.critic_optimizer.step()

            if full_observable_critic_flag:
                if isinstance(self.actors, list):
                    action_i = action_batch.clone()
                    one_agent_batch_act = self.actors[agent]([stacked_elem_0_combine[:, agent, :], stacked_elem_1_combine[:, agent, :], stacked_elem_2_combine[:, agent, :]])
                    action_i[:, agent, :] = one_agent_batch_act
                else:
                    # for one model
                    action_i = []
                    for i in range(self.n_agents):
                        one_agent_act = self.actors(
                            [stacked_elem_0_combine[:,i,:], stacked_elem_1_combine[:,i,:], stacked_elem_2_combine[:,i,:]])
                        action_i.append(one_agent_act)
                    action_i = torch.stack(action_i, dim=1)

            else:
                if use_GRU_flag:
                    action_i = self.actors([stacked_elem_0, stacked_elem_1], agents_cur_hidden_state)[0]
                elif use_selfATT_with_radar or use_allNeigh_wRadar:
                    action_i = self.actors([stacked_elem_0, stacked_elem_1, stacked_elem_2])
                else:
                    action_i = self.actors([stacked_elem_0, stacked_elem_1])
                ac = action_i.squeeze(0)  # replace the actor from self.actors[agent] into action batch


            # actor_loss = -self.critics[agent](whole_state, whole_action_action_replaced, whole_hs).mean()
            # actor_loss = 3-self.critics[agent](whole_state, whole_action_action_replaced, whole_agent_combine_gru).mean()
            if full_observable_critic_flag:
                actor_loss = -self.critics([stacked_elem_0_combine, stacked_elem_1_combine, stacked_elem_2_combine], action_i).mean()
                # actor_loss = - self.critics[agent]([stacked_elem_0_combine, stacked_elem_1_combine], combine_action_action_replaced).mean()
            else:
                # actor_loss = 3 - self.critics[agent]([stacked_elem_0[:, agent, :], stacked_elem_1[:, agent, :]],
                #                                      ac[:, agent, :]).mean()
                # actor_loss = - self.critics[agent]([stacked_elem_0[:, agent, :], stacked_elem_1[:, agent, :]],
                #                                      ac[:, agent, :]).mean()
                if use_GRU_flag:
                    actor_loss = - self.critics([stacked_elem_0, stacked_elem_1], ac, agents_cur_hidden_state)[0].mean()
                elif use_selfATT_with_radar or use_allNeigh_wRadar:
                    actor_loss = - self.critics([stacked_elem_0, stacked_elem_1, stacked_elem_2], ac).mean()
                else:
                    actor_loss = - self.critics([stacked_elem_0, stacked_elem_1], ac).mean()

            # actor_update_time = time.time()
            # actor_loss = -self.critics[agent](stacked_elem_0[:,agent,:], ac[:, agent, :], agents_cur_hidden_state[:, agent, :])[0].mean()
            if transfer_learning:
                if i_episode > 10000:
                    self.actor_optimizer[agent].zero_grad()
                    actor_loss.backward()
                    # self.has_gradients(self.actors[agent], agent, wandb)  # Replace with your actor network variable
                    self.actor_optimizer[agent].step()
            else:
                if isinstance(self.actors, list):
                    # self.actor_optimizer[agent].zero_grad()
                    self.actor_optimizer[agent].zero_grad()
                    actor_loss.backward()
                    # self.has_gradients(self.actors[agent], agent, wandb)  # Replace with your actor network variable
                    # self.actor_optimizer[agent].step()
                    self.actor_optimizer[agent].step()
                else:
                    # self.actor_optimizer[agent].zero_grad()
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    # self.has_gradients(self.actors[agent], agent, wandb)  # Replace with your actor network variable
                    # self.actor_optimizer[agent].step()
                    self.actor_optimizer.step()
            # end_actor_update_time = (time.time() - actor_update_time)*1000
            # print("loop {} actor update time used is {} ms".format(agent, end_actor_update_time))
            c_loss.append(loss_Q)
            a_loss.append(actor_loss)

        if isinstance(self.actors, list):
            for agent in range(var_iteration):
                if i_episode % UPDATE_EVERY == 0:  # perform a soft update at each step of an episode.
                    soft_update(self.critics_target, self.critics, self.tau)
                    soft_update(self.actors_target[agent], self.actors[agent], self.tau)
        else:
            if i_episode % UPDATE_EVERY == 0:  # perform a soft update at each step of an episode.
                soft_update(self.critics_target, self.critics, self.tau)
                soft_update(self.actors_target, self.actors, self.tau)

        return c_loss, a_loss, single_eps_critic_cal_record

    def has_gradients(self, model, agent, wandb=None):
        for name, param in model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"Numerical instability detected in {name}")
            if param.grad is None:
                print(f"No gradient for {name}")
            else:
                # print(f"Gradient for {name} is {param.grad.norm()}")
                # wandb.log(
                #     {f"actor/{agent}_/{name}_gradients_histogram": wandb.Histogram(param.grad.cpu().detach().numpy())})
                wandb.log({name: float(param.grad.norm())})
                # wandb.log({'agent' + str(idx): wandb.Histogram(param.grad.cpu().detach().numpy())})

    def choose_action(self, state, cur_total_step, cur_episode, step, mini_noise_eps, noise_start_level, actor_hiddens, use_allNeigh_wRadar, use_selfATT_with_radar, own_obs_only, noisy=True, use_GRU_flag=False):

        obs = torch.from_numpy(np.stack(state[0])).to(self.device)
        obs_full_nei = torch.from_numpy(np.stack(state[1])).to(self.device)
        obs_grid = torch.from_numpy(np.stack(state[2])).to(self.device)

        noise_value = np.zeros(2)

        # if len(gru_history) < self.args.gru_history_length:
        #     # Append zero arrays to fill the gru_history
        #     for _ in range(self.args.gru_history_length - len(gru_history)):
        #         zero_array = np.zeros((self.n_agents, self.n_actor_dim[0]))
        #         gru_history.append(zero_array)
        # gru_history_input = np.array(gru_history)
        # gru_history_input = np.expand_dims(gru_history_input, axis=0)

        actions = torch.zeros(self.n_agents, self.n_actions, dtype=torch.float64)
        # act_hn = torch.zeros(self.n_agents, self.n_actions)
        if use_GRU_flag:
            act_hn = torch.zeros(self.n_agents, self.actors.rnn_hidden_dim)
        else:
            act_hn = torch.zeros(self.n_agents, self.n_actions)
        if self.device.type == 'cuda':
            FloatTensor = torch.cuda.FloatTensor
        else:
            FloatTensor = torch.FloatTensor
        # this for loop used to decrease noise level for all agents before taking any action
        # gru_history_input = torch.FloatTensor(gru_history_input).to(device)  # batch x seq_length x no_agent x feature_length
        if isinstance(actor_hiddens, list):
            gru_history_input = torch.FloatTensor(actor_hiddens).unsqueeze(0).to(self.device)  # batch x no_agent x feature_length
        else:
            gru_history_input = actor_hiddens
        for i in range(self.n_agents):
            self.var[i] = self.get_custom_linear_scaling_factor(cur_episode, mini_noise_eps, noise_start_level)  # self.var[i] will decrease as the episode increase
            # self.var[i] = self.exponential_decay_variance(cur_episode, mini_noise_eps, noise_start_level)  # self.var[i] will decrease as the episode increase
            # self.var[i] = self.linear_decay(episode, eps_end, noise_start_level)  # self.var[i] will decrease as the episode increase

        for i in range(self.n_agents):
            # sb = obs[i].detach()
            # sb_grid = obs_grid[i].detach()
            # sb_surAgent = all_obs_surAgent[i].detach()
            sb = obs[i]
            sb_full_nei = obs_full_nei[i]
            sb_grid = obs_grid[i]
            # sb_surAgent = all_obs_surAgent[i]
            # act = self.actors[i]([sb.unsqueeze(0), sb_grid.unsqueeze(0), sb_surAgent.unsqueeze(0)]).squeeze()
            # act = self.actors[i]([sb.unsqueeze(0), sb_surAgent.unsqueeze(0)]).squeeze()
            # act, hn = self.actors[i](sb.unsqueeze(0), gru_history_input[:,:,i,:])
            # act, hn = self.actors[i](sb.unsqueeze(0), gru_history_input[:, i, :])
            # act = self.actors[i]([sb.unsqueeze(0), sb_grid.unsqueeze(0)])
            # using one model
            if use_GRU_flag:
                act, hn = self.actors([sb.unsqueeze(0), sb_grid.unsqueeze(0)], gru_history_input[:, i, :])
            elif use_selfATT_with_radar or use_allNeigh_wRadar:
                if own_obs_only:
                    act = self.actors([sb.unsqueeze(0)])
                else:
                    if isinstance(self.actors, list):
                        act = self.actors[i]([sb.unsqueeze(0), sb_full_nei.unsqueeze(0), sb_grid.unsqueeze(0)])
                    else:
                        act = self.actors([sb.unsqueeze(0), sb_full_nei.unsqueeze(0), sb_grid.unsqueeze(0)])
            else:
                act = self.actors([sb.unsqueeze(0), sb_full_nei.unsqueeze(0)])
            # act = self.actors([sb.unsqueeze(0), sb_grid.unsqueeze(0)])
            if noisy:
                noise_value = np.random.randn(2) * self.var[i]
                # noise_value = self.OU_noise.sample(self.var[i])  # OU noise
                act = act + torch.from_numpy(noise_value).to(self.device)
                # print("Episode {}, agent {}, noise level is {}".format(episode, i, self.var[i]))
                act = torch.clamp(act, -1.0, 1.0)  # when using stochastic policy, we are not require to clamp again.

            actions[i, :] = act
            if use_GRU_flag:
                act_hn[i, :] = hn
            else:
                act_hn[i, :] = torch.zeros(1, self.n_actions)
            # act_hn[i, :] = torch.zeros(1, self.n_actions)
        self.steps_done += 1
        # ------------- end of MADDPG_test_181123_10_10_54 version noise -------------------

        # obs = torch.from_numpy(np.stack(state[0])).float().to(device)
        # obs_grid = torch.from_numpy(np.stack(state[1])).float().to(device)
        # noise_value = np.zeros(2)
        # all_obs_surAgent = []
        # for each_agent_sur in state[2]:
        #     try:
        #         each_obs_surAgent = np.squeeze(np.array(each_agent_sur), axis=1)
        #         all_obs_surAgent.append(torch.from_numpy(each_obs_surAgent).float().to(device))
        #     except:
        #         print("pause and check")
        #
        # # obs_surAgent = torch.from_numpy(np.stack(state[2])).float().to(device)
        #
        # actions = torch.zeros(self.n_agents, self.n_actions)
        # FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        # # this for loop used to decrease noise level for all agents before taking any action
        # # for i in range(self.n_agents):
        # #     self.var[i] = self.get_custom_linear_scaling_factor(episode, eps_end, noise_start_level)  # self.var[i] will decrease as the episode increase
        #
        # for i in range(self.n_agents):
        #     # sb = obs[i].detach()
        #     # sb_grid = obs_grid[i].detach()
        #     # sb_surAgent = all_obs_surAgent[i].detach()
        #     sb = obs[i]
        #     sb_grid = obs_grid[i]
        #     sb_surAgent = all_obs_surAgent[i]
        #     # act = self.actors[i]([sb.unsqueeze(0), sb_grid.unsqueeze(0), sb_surAgent.unsqueeze(0)]).squeeze()
        #     # act = self.actors[i]([sb.unsqueeze(0), sb_surAgent.unsqueeze(0)]).squeeze()
        #     act = self.actors[i]([sb.unsqueeze(0)]).squeeze()
        #     if noisy:
        #         noise_value = np.random.randn(2) * self.var[i]
        #         act += torch.from_numpy(noise_value).type(FloatTensor)
        #         # print("Episode {}, agent {}, noise level is {}".format(episode, i, self.var[i]))
        #         act = torch.clamp(act, -1.0, 1.0)  # when using stochastic policy, we are not require to clamp again.
        #
        #     actions[i, :] = act
        #
        # for i in range(self.n_agents):
        #     if self.var[i] > 0.05:  # noise decrease at every step instead of every episode.
        #         self.var[i] = self.var[i] * 0.999998
        # self.steps_done += 1

        return actions.data.cpu().numpy(), noise_value, gru_history_input.squeeze(0).data.cpu(), act_hn.data.cpu()  # NOTE: tensor.data.cpu() is to make the tensor's "is_leaf" = True, this also prevent the error message on line "retain_graph=True"
        # return actions.data.cpu().numpy(), noise_value, gru_history_input.squeeze(0).data.cpu()  # NOTE: tensor.data.cpu() is to make the tensor's "is_leaf" = True, this also prevent the error message on line "retain_graph=True"
        # return actions.data.cpu().numpy(), noise_value

    def exponential_decay_variance(self, episode, eps_end, start_scale=1, end_scale=0.03):
        if episode <= eps_end:
            decay_rate = -np.log(end_scale / (start_scale - end_scale)) / eps_end
            final_variance = end_scale + (start_scale - eps_end) * np.exp(-decay_rate * episode)
        else:
            final_variance = end_scale
        return final_variance

    # def get_custom_linear_scaling_factor(self, episode, eps_end, start_scale=1, end_scale=0.03):
    def get_custom_linear_scaling_factor(self, episode, eps_end, start_scale=1, end_scale=0):
        # Calculate the slope of the linear decrease only up to eps_end
        if episode <= eps_end:
            slope = (end_scale - start_scale) / (eps_end - 1)
            current_scale = start_scale + slope * (episode - 1)
        else:
            current_scale = end_scale
        return current_scale

    def linear_decay(self, cur_total_step, total_training_steps, start_scale=1, end_scale=0.03):
        current_scale = start_scale
        # Calculate the slope of the linear decrease only up to eps_end
        if cur_total_step == 0:
            return current_scale
        if current_scale > end_scale:
            eps_delta = (start_scale - end_scale) / total_training_steps
            current_scale = start_scale - eps_delta
        else:
            current_scale = end_scale
        return current_scale