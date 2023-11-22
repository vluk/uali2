import time
from copy import deepcopy

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

import ray

from tetris.nnet import NNet
from replay_buffer import ReplayBuffer

kl = nn.KLDivLoss(reduction="batchmean")
mse = nn.MSELoss()

eps = 0.0001

@ray.remote(num_gpus=0.15)
class Trainer():
    def __init__(self, initial_state_dict):
        self.nnet = NNet().cuda()
        self.nnet.load_state_dict(initial_state_dict)
        self.optimizer = optim.Adam(self.nnet.parameters(), lr=0.00001)

    def train(self, replay_buffer: ReplayBuffer):
        print("waiting...")
        while ray.get(replay_buffer.get_iterations.remote()) < 10000:
            time.sleep(1)

        print("starting training")
        self.nnet.train()

        i = 0
        next_batch = replay_buffer.get_batch.remote()
        while True:
            batch = ray.get(next_batch)
            next_batch = replay_buffer.get_batch.remote()
            self.train_batch(batch)
            if i % 5000 == 0:
                replay_buffer.save_state_dict.remote(deepcopy(self.nnet.state_dict()))
                path = f"checkpoints/{i:09}_checkpoint.pth"
                torch.save(self.nnet.state_dict(), path)
                print(f"Saving checkpoint to {path}")
            i += 1

    def train_batch(self, batch):
        obs, pi, v = batch

        pi = torch.FloatTensor(pi.copy()).contiguous().cuda()
        v = torch.FloatTensor(v.copy()).contiguous().cuda()

        # compute output
        with torch.autocast("cuda"):
            out_logits, out_v = self.nnet(obs)
            l_pi = kl(F.log_softmax(out_logits.reshape(-1, 1600), dim=1), pi.reshape(-1, 1600))
            l_v = mse(out_v, v)
            total_loss = l_pi + l_v

        # compute gradient and do SGD step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
