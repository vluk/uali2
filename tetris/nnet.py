import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

NUM_CHANNELS = 32

class ResNet(nn.Module):
    def __init__(self):
        # game params
        self.board_x, self.board_y = 10, 6
        self.action_size = 8 * 10 * 6

        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, NUM_CHANNELS, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(NUM_CHANNELS, NUM_CHANNELS, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(NUM_CHANNELS, NUM_CHANNELS, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(NUM_CHANNELS, NUM_CHANNELS, 3, stride=1, padding=1)

        self.qfc1 = nn.Linear(14*7, 128)
        self.qfc2 = nn.Linear(128, 128)

        self.cfc1 = nn.Linear(6*20, 128)
        self.cfc2 = nn.Linear(128, 128)


        self.fc1 = nn.Linear(2048, 2048)
        self.fc2 = nn.Linear(2048 + 128, 2048)
        self.fc3 = nn.Linear(4096, 2048)
        self.fc4 = nn.Linear(2048, self.action_size)
        self.fc5 = nn.Linear(2048, 1)
    
    def body(self, b, q, c):

        return s

    def forward(self, b1, b2, q1, q2, c1, c2):
        s1 = b1.view(-1, 1, 10, 6)
        q1 = q1.view(-1, 49)
        c1 = c1.view(-1, 60)

        s2 = b1.view(-1, 1, 10, 6)
        q2 = q1.view(-1, 49)
        c2 = c1.view(-1, 60)

        s1 = F.relu(self.conv1(s1))
        s1 = F.relu(self.conv2(s1))
        q1 = F.relu(self.qfc1(q1))
        q1 = F.relu(self.qfc2(q1))
        c1 = F.relu(self.cfc1(c1))
        c1 = F.relu(self.cfc2(c1))

        s1 = s1.view(-1, 1920)
        s1 = torch.cat((s1, q1), 1)
        s1 = F.relu(self.fc1(s1))
        s1 = torch.cat((s1, c1), 1)
        s1 = F.relu(self.fc2(s1))

        s2 = F.relu(self.conv1(s2))
        s2 = F.relu(self.conv2(s2))
        q2 = F.relu(self.qfc1(q2))
        q2 = F.relu(self.qfc2(q2))
        c2 = F.relu(self.cfc1(c2))
        c2 = F.relu(self.cfc2(c2))

        s2 = s2.view(-1, 1920)
        s2 = torch.cat((s2, q2), 1)
        s2 = F.relu(self.fc1(s2))
        s2 = torch.cat((s2, c2), 1)
        s2 = F.relu(self.fc2(s2))

        s = torch.cat((s1, s2), 1)

        s = F.relu(self.fc3(s))

        pi = self.fc4(s)                                                                         # batch_size x action_size
        v = self.fc5(s)                                                                          # batch_size x 1

        return pi, v

class NNetWrapper():
    def __init__(self):
        self.nnet = ResNet()
        self.board_x, self.board_y = 10, 6
        self.action_size = (8, 10, 6)

        self.nnet.cuda()
        self.nnet.eval()

    def train(self, replay_buffer, iters=10000):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        optimizer = optim.Adam(self.nnet.parameters())
        kl = nn.KLDivLoss()
        mse = nn.MSELoss()

        self.nnet.train()
        for i in range(iters):

            obs, pi, v = replay_buffer.get_batch() 

            obs = (torch.FloatTensor(i).contiguous().cuda() for i in obs)
            pi = torch.FloatTensor(pi).contiguous().cuda()
            v = torch.FloatTensor(v).contiguous().cuda()

            b1, b2, q1, q2, c1, c2 = obs

            # compute output
            out_logits, out_v = self.nnet(b1, b2, q1, q2, c1, c2)
            l_pi = kl(F.log_softmax(out_logits, dim=1), F.softmax(pi, dim=1))
            l_v = mse(out_v, v)
            total_loss = l_pi + l_v

            # compute gradient and do SGD step
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        self.nnet.eval()

    def predict(self, obs):
        """
        board: np array with board
        """
        # timing

        # preparing input
        obs = (torch.FloatTensor(i).contiguous().cuda().unsqueeze(0) for i in obs)
        b1, b2, q1, q2, c1, c2 = obs

        start = time.time()

        pi, v = self.nnet(b1, b2, q1, q2, c1, c2)

        print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        pi = pi.view(8, 10, 6)

        pi, v = torch.exp(pi).data.cpu().numpy(), v.data.cpu().numpy()[0][0]

        return pi, v