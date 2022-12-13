import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
import time

NUM_CHANNELS = 32

class NNet(nn.Module):
    def __init__(self):
        # game params
        self.action_size = 8 * 10 * 6

        super(NNet, self).__init__()
        self.conv1 = nn.Conv2d(1, NUM_CHANNELS, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(NUM_CHANNELS, NUM_CHANNELS, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(NUM_CHANNELS, NUM_CHANNELS, 3, stride=1, padding=1)

        self.tfc_1 = nn.Linear(109, 128)

        self.tfc_2 = nn.Linear(128, 128)

        self.fc1 = nn.Linear(2048, 2048)
        self.fc2 = nn.Linear(2048, 2048)

        self.fc3 = nn.Linear(4096, 2048)
        self.fc4 = nn.Linear(2048, self.action_size)
        self.fc5 = nn.Linear(2048, 1)

    @autocast("cuda")
    def forward(self, b1, b2, q1, q2, c1, c2):
        s1 = b1.view(-1, 1, 10, 6)
        t1 = torch.cat((q1.view(-1, 49), c1.view(-1, 60)), 1)

        s1 = F.relu(self.conv1(s1))
        s1 = F.relu(self.conv2(s1))
        s1 = F.relu(self.conv3(s1))

        t1 = F.relu(self.tfc_1(t1))
        t1 = F.relu(self.tfc_2(t1))

        s1 = s1.view(-1, 1920)
        s1 = torch.cat((s1, t1), 1)
        s1 = F.relu(self.fc1(s1))

        s2 = b2.view(-1, 1, 10, 6)
        s2 = F.relu(self.conv1(s2))
        s2 = s2.view(-1, 1920)
        t2 = torch.cat((q2.view(-1, 49), c2.view(-1, 60)), 1)
        t2 = F.relu(self.tfc_1(t2))
        s2 = torch.cat((s2, t2), 1)
        s2 = F.relu(self.fc2(s2))

        s = torch.cat((s1, s2), 1)

        s = F.relu(self.fc3(s))

        pi = self.fc4(s)                                                                         # batch_size x action_size
        v = self.fc5(s)                                                                          # batch_size x 1

        return pi, v

    def predict(self, obs):
        """
        board: np array with board
        """
        # timing

        # preparing input
        obs = (torch.FloatTensor(i).contiguous().cuda().unsqueeze(0) for i in obs)
        b1, b2, q1, q2, c1, c2 = obs

        pi, v = self.forward(b1, b2, q1, q2, c1, c2)

        pi = pi.view(8, 10, 6)

        pi, v = torch.exp(pi).data.cpu().numpy(), v.data.cpu().numpy()[0][0]

        return pi, v