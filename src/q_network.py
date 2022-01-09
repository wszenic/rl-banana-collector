import torch
import torch.nn as nn
import torch.nn.functional as F

from config import MID_1_SIZE, MID_2_SIZE


class QNetwork(nn.Module):

    def __init__(self, state_size: int, action_size: int, seed=42):
        # dueling q network
        super().__init__()
        self.seed = torch.manual_seed(seed)

        self.linear_1_advantage = nn.Linear(state_size, MID_1_SIZE)
        self.linear_2_advantage = nn.Linear(MID_1_SIZE, MID_2_SIZE)
        self.linear_3_advantage = nn.Linear(MID_2_SIZE, action_size)

        self.linear_1_state = nn.Linear(state_size, MID_1_SIZE)
        self.linear_2_state = nn.Linear(MID_1_SIZE, MID_2_SIZE)
        self.linear_3_state = nn.Linear(MID_2_SIZE, 1)

    def forward(self, x):
        x = torch.Tensor(x)

        adv = F.relu(self.linear_1_advantage(x))
        adv = F.relu(self.linear_2_advantage(adv))
        adv = F.relu(self.linear_3_advantage(adv))

        val = F.relu(self.linear_1_state(x))
        val = F.relu(self.linear_2_state(val))
        val = F.relu(self.linear_3_state(val))

        res = val + adv - torch.mean(adv)

        return res



