import random
from collections import namedtuple

import numpy as np
import torch

from src.config import UPDATE_INTERVAL, BATCH_SIZE, BUFFER_SIZE, GAMMA, TAU, LEARNING_RATE, CHECKPOINT_SAVE_PATH
from src.memory import ReplayBuffer
from src.q_network import QNetwork


class Agent:

    def __init__(self, state_size: int, action_size: int, read_saved_model=False):

        self.state_size = state_size
        self.action_size = action_size

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

        self.q_network_local = QNetwork(state_size, action_size).to(self.device)
        self.q_network_target = QNetwork(state_size, action_size).to(self.device)

        if read_saved_model:
            saved_model = torch.load(CHECKPOINT_SAVE_PATH)
            self.q_network_local.load_state_dict(saved_model)
            self.q_network_target.load_state_dict(saved_model)

        self.optimizer = torch.optim.Adam(self.q_network_local.parameters(), lr=LEARNING_RATE)

        self.gamma = GAMMA
        self.tau = TAU

        self._step = 0
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, self.device)
        self.env_feedback = namedtuple('env_feedback', ('state', 'action', 'reward', 'next_state', 'done'))

    def step(self, env_data):
        self.memory.add(
            state=env_data.state,
            action=env_data.action,
            reward=env_data.reward,
            next_state=env_data.next_state,
            done=env_data.done,
        )

        self._step = (self._step + 1) % UPDATE_INTERVAL
        if self._step == 0:
            if len(self.memory) > BATCH_SIZE:
                experience_replay = self.memory.sample()

                self.learn(self.env_feedback(*experience_replay))

    def act(self, state, eps):
        state = torch.from_numpy(state).float().unsqueeze(0)
        action_values = self.__get_state_action_values(state)
        return self.__get_epsilon_greedy_action(action_values, eps)

    def learn(self, env_data):
        q_next_targets = self.q_network_target(torch.Tensor(env_data.next_state)).detach().max(1)[0].unsqueeze(1)
        q_target = env_data.reward + (self.gamma * q_next_targets * (1 - env_data.done))

        q_expected = self.q_network_local(env_data.state).gather(1, env_data.action)
        self.__torch_learn(q_expected, q_target)

    def __torch_learn(self, q_expected, q_target):
        loss = torch.nn.functional.mse_loss(q_expected, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.__soft_update(self.q_network_local, self.q_network_target)

    def __get_state_action_values(self, state):
        self.q_network_local.eval()
        with torch.no_grad():
            actions_values = self.q_network_local(state)
        self.q_network_local.train()

        return actions_values

    def __get_epsilon_greedy_action(self, action_values, eps):
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def __soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)