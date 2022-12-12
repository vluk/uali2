import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

NUM_CHANNELS = 16

class ResNet(nn.Module):
    def __init__(self):
        # game params
        self.board_x, self.board_y = 28, 14
        self.action_size = 8 * 10 * 6

        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, NUM_CHANNELS, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(NUM_CHANNELS, NUM_CHANNELS, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(NUM_CHANNELS, NUM_CHANNELS, 3, stride=1)
        self.conv4 = nn.Conv2d(NUM_CHANNELS, NUM_CHANNELS, 3, stride=1)

        self.bn1 = nn.BatchNorm2d(NUM_CHANNELS)
        self.bn2 = nn.BatchNorm2d(NUM_CHANNELS)
        self.bn3 = nn.BatchNorm2d(NUM_CHANNELS)
        self.bn4 = nn.BatchNorm2d(NUM_CHANNELS)

        self.fc1 = nn.Linear(NUM_CHANNELS*(self.board_x-4)*(self.board_y-4), 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, self.action_size)

        self.fc4 = nn.Linear(512, 1)

    def forward(self, s):
        #                                                           s: batch_size x board_x x board_y
        s = s.view(-1, 1, self.board_x, self.board_y)                # batch_size x 1 x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn2(self.conv2(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn3(self.conv3(s)))                          # batch_size x num_channels x (board_x-2) x (board_y-2)
        s = F.relu(self.bn4(self.conv4(s)))                          # batch_size x num_channels x (board_x-4) x (board_y-4)
        s = s.view(-1, NUM_CHANNELS*(self.board_x-4)*(self.board_y-4))

        s = F.relu(self.fc_bn1(self.fc1(s)))  # batch_size x 1024
        s = F.relu(self.fc_bn2(self.fc2(s)))  # batch_size x 512

        pi = self.fc3(s)                                                                         # batch_size x action_size
        v = self.fc4(s)                                                                          # batch_size x 1

        return pi, v

class NNetWrapper():
    def __init__(self):
        self.nnet = ResNet()
        self.board_x, self.board_y = 28, 14
        self.action_size = (8, 10, 6)

        self.nnet.cuda()

    def train(self, replay_buffer, iters=1000):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        optimizer = optim.Adam(self.nnet.parameters())
        kl = nn.KLDivLoss()
        mse = nn.MSELoss()

        self.nnet.train()
        for i in range(iters):

            obs, pi, v = replay_buffer.get_batch() 
            obs = torch.FloatTensor(obs).contiguous().cuda()
            pi = torch.FloatTensor(pi).contiguous().cuda()
            v = torch.FloatTensor(v).contiguous().cuda()

            # compute output
            out_logits, out_v = self.nnet(obs)
            l_pi = kl(F.log_softmax(out_logits, dim=1), F.softmax(pi, dim=1))
            l_v = mse(out_v, v)
            total_loss = l_pi + l_v

            # compute gradient and do SGD step
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

    def predict(self, board):
        """
        board: np array with board
        """
        # timing

        # preparing input
        board = torch.FloatTensor(board.astype(np.float64)).contiguous().cuda()
        board = board.view(1, 28, 14)
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)
        pi = pi.view(1, 8, 10, 6)

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0][0]