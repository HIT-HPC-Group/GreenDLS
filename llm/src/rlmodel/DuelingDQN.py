# -*- coding: utf-8 -*-
# 2022.8.23
# power capping in V100S
# min power limit 100W
# max power limit 250W
# default power limit 250W
# Memory 1107MHz
# core Frequency 1597HMz ~ 135HMz step:7/8HMz
import math
import os
import torch
import numpy as np
import torch.nn as nn
import pynvml
import glva
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random
import torch.optim as optim
from rlmodel.rl_utils import *

# # 强化学习 State：(1)GPU频率，(2)GPU利用率，(3)显存利用率，(4)实时功耗，(5)实时温度，
# GPU_LABELS = (
#                'UTIL_GPU'
#               , 'UTIL_MEM'
#               , 'POWER'
#               , 'TEMP'
#               )
# MINS = { 'UTIL_GPU': 0, 'UTIL_MEM': 0, 'POWER': 25, 'TEMP': 30}
# MAXS = { 'UTIL_GPU': 100, 'UTIL_MEM': 100, 'POWER': 250, 'TEMP': 100}
# BUCKETS = { 'UTIL_GPU': 20, 'UTIL_MEM': 20, 'POWER': 20, 'TEMP': 30}
# gpu_num_buckets = np.array([BUCKETS[k] for k in GPU_LABELS], dtype=np.double)
# #gpu frequency as one of state
# max_clock = 1597
# min_clock = 135
# clock=max_clock
# CLOCKS_GPU =[]
# while clock > min_clock:
#     CLOCKS_GPU.append(clock)
#     clock = clock-7
#     CLOCKS_GPU.append(clock)
#     clock = clock - 8
# # print(CLOCKS_GPU)
# clock_gpu_bucket = { CLOCKS_GPU[i]: i for i in range(len(CLOCKS_GPU))}
# POWER_IN_STATE = 0  # 是否将功率上限作为一种 State

# #action
# gpu_limit = 1590
# max_freq = 1590
# min_freq = 765
# clock = max_freq
# GPU = []
# while clock > min_freq:
#     GPU.append(clock)
#     clock = clock-15
# gpu_to_bucket = {GPU[i]: i for i in range(len(GPU))}


# # result_gr=open(save_path+"_rl_gpu.txt","w")
# pynvml.nvmlInit() 
# handle = pynvml.nvmlDeviceGetHandleByIndex(0)
# def state():
#     gpu_limit=pynvml.nvmlDeviceGetMaxClockInfo( handle,0)
#     clock_gpu=pynvml.nvmlDeviceGetClockInfo( handle,0)
#     util_gpu=pynvml.nvmlDeviceGetUtilizationRates( handle).gpu
#     memory_info=pynvml.nvmlDeviceGetMemoryInfo( handle)
#     util_memory=memory_info.used/memory_info.total
#     power_gpu=pynvml.nvmlDeviceGetPowerUsage( handle)/1000
#     temp=pynvml.nvmlDeviceGetTemperature( handle, 0)
#     stats = {
#     'GPUL': gpu_limit,
#     'CLOCKS_GPU': clock_gpu,
#     'UTIL_GPU': util_gpu,
#     'UTIL_MEM': util_memory,
#     'POWER': power_gpu,
#     'TEMP': temp}
#     # GPU states
#     # result_gr.write("****************GPU Status:"+str(stats)+"\n")
#     gpu_all_mins = np.array([MINS[k] for k in GPU_LABELS], dtype=np.double)
#     gpu_all_maxs = np.array([MAXS[k] for k in GPU_LABELS], dtype=np.double)
#     gpu_num_buckets = np.array([BUCKETS[k] for k in GPU_LABELS], dtype=np.double)
#     gpu_widths = np.divide(np.array(gpu_all_maxs) - np.array(gpu_all_mins), gpu_num_buckets)  # divide /

#     gpu_raw_no_pow = [stats[k] for k in GPU_LABELS]  # wym modify
#     gpu_raw_no_pow = np.clip(gpu_raw_no_pow, gpu_all_mins, gpu_all_maxs)  # clip set data at range(min max)

#     gpu_raw_floored = gpu_raw_no_pow - gpu_all_mins
#     gpu_state = np.divide(gpu_raw_floored, gpu_widths)
#     gpu_state = np.clip(gpu_state, 0, gpu_num_buckets - 1)

#     gpu_state = np.append(gpu_state, [clock_gpu_bucket[stats['CLOCKS_GPU']]]) # Add mem frequency index to end of state:
#     if POWER_IN_STATE:
#         # Add power cap index to end of state:
#         gpu_state = np.append(gpu_state, [gpu_to_bucket[stats['GPUL']]])

#     # Convert floats to integer bucket indices and return:
#     gpu_state = [int(x) for x in gpu_state]
#     #gpu state(1)GPU频率，(2)GPU利用率，(3)显存利用率，(4)实时功耗，(4)实时温度，#(6)显存频率 (7)可能还有功率上限 或者 频率上限
#     glva.result_txt.write("gpu state :{}\n stats : {}\n".format(gpu_state,stats))
#     return gpu_state, stats

# STEPS = 22000
STEPS = 8000
minimal_size = 10000
class DuelingDeepQNetwork(nn.Module):
    def __init__(self, alpha, state_dim, action_dim, fc1_dim, fc2_dim,device):
        super(DuelingDeepQNetwork, self).__init__()
        self.input_size=state_dim
        # self.fc1 = nn.Linear(state_dim, fc1_dim)
        # self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.input_size, fc1_dim),
            nn.ReLU(),
            nn.Linear(fc1_dim, fc2_dim),
            # nn.ReLU(),
            # nn.Linear(fc2_dim, fc2_dim),
            nn.Tanh()
            )
        self.V = nn.Linear(fc2_dim, 1)
        self.A = nn.Linear(fc2_dim, action_dim)
        self.device=device
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.to(self.device)

    def forward(self, state):
        # x = torch.relu(self.fc1(state))
        # x = torch.relu(self.fc2(x))
        x = self.mlp(state)
        V = self.V(x)
        A = self.A(x)

        return V, A

    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file, _use_new_zipfile_serialization=False)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))


class DuelingDQN:
    def __init__(self, alpha, state_dim, action_dim, fc1_dim, fc2_dim,device,
                 gamma=0.99, tau=0.005, epsilon=1.0, eps_end=0.1, eps_dec=1e-3,
                 max_size=1000000, batch_size=256):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.device=device
        self.action_space = [i for i in range(action_dim)]

        self.q_eval = DuelingDeepQNetwork(alpha=alpha, state_dim=state_dim, action_dim=action_dim,
                                          fc1_dim=fc1_dim, fc2_dim=fc2_dim,device=device)
        self.q_target = DuelingDeepQNetwork(alpha=alpha, state_dim=state_dim, action_dim=action_dim,
                                            fc1_dim=fc1_dim, fc2_dim=fc2_dim,device=device)

        self.memory = ReplayBuffer(state_dim=state_dim, action_dim=action_dim,
                                   max_size=max_size, batch_size=batch_size)

        self.update_network_parameters(tau=1.0)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        for q_target_params, q_eval_params in zip(self.q_target.parameters(), self.q_eval.parameters()):
            q_target_params.data.copy_(tau * q_eval_params + (1 - tau) * q_target_params)

    def remember(self, state, action, reward, stata_, done):
        self.memory.store_transition(state, action, reward, stata_, done)
        

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min

    def choose_action(self, observation,is_trained=False):
        if np.random.random() < self.epsilon :
            action = np.random.choice(self.action_space)
            batchsize=glva.action_GPU[action][0]
            glva.result_txt.write("=====================random action\n")
            if glva.sendspeed[0]==2:
                while( batchsize <=4 or batchsize>=15):
                    action = np.random.choice(self.action_space)
                    batchsize=glva.action_GPU[action][0]
            elif glva.sendspeed[0]==4:
                while( batchsize <=8 or batchsize>28):
                    action = np.random.choice(self.action_space)
                    batchsize=glva.action_GPU[action][0]
            else:
                while(batchsize<=8):
                    action = np.random.choice(self.action_space)
                    batchsize=glva.action_GPU[action][0]
        else:
            glva.result_txt.write("=====================argmax action\n")
            state = torch.tensor([observation], dtype=torch.float).to(self.device)
            V, A = self.q_eval.forward(state)
            q_value=V + A - torch.mean(A, dim=-1, keepdim=True)
            action = torch.argmax(q_value).item()
        return action
    def get_QValue(self,observation):
        state = torch.tensor([observation], dtype=torch.float).to(self.device)
        V, A = self.q_eval.forward(state)
        q_value=V + A - torch.mean(A, dim=-1, keepdim=True)
        return torch.argmax(q_value).item()
    def learn(self):
        if not self.memory.ready():
            return

        states, actions, rewards, next_states, terminals = self.memory.sample_buffer()
        batch_idx = torch.arange(self.batch_size, dtype=torch.long).to(self.device)
        states_tensor = torch.tensor(states, dtype=torch.float).to(self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float).to(self.device)
        next_states_tensor = torch.tensor(next_states, dtype=torch.float).to(self.device)
        terminals_tensor = torch.tensor(terminals).to(self.device)

        with torch.no_grad():
            V_, A_ = self.q_target.forward(next_states_tensor)
            q_ = V_ + A_ - torch.mean(A_, dim=-1, keepdim=True)
            q_[terminals_tensor] = 0.0
            target = rewards_tensor + self.gamma * torch.max(q_, dim=-1)[0]
        V, A = self.q_eval.forward(states_tensor)
        q = (V + A - torch.mean(A, dim=-1, keepdim=True))[batch_idx, actions_tensor]
        # q = (V + A - torch.mean(A, dim=-1, keepdim=True)).gather(1, actions_tensor)
        loss = F.mse_loss(q, target.detach())
        self.q_eval.optimizer.zero_grad()
        loss.backward()
        self.q_eval.optimizer.step()

        self.update_network_parameters()
        self.decrement_epsilon()
        return loss


