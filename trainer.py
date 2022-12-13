import time
from copy import deepcopy

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
import ray

from tetris.nnet import NNet
from mcts import MCTS
from tetris.tetris_game import Game
from replay_buffer import ReplayBuffer

kl = nn.KLDivLoss()
mse = nn.MSELoss()

@ray.remote(num_gpus=0.15)
class Trainer():
    def __init__(self, initial_state_dict):
        self.nnet = NNet().cuda()
        self.nnet.load_state_dict(initial_state_dict)
        self.optimizer = optim.Adam(self.nnet.parameters(), lr=0.00005)

    def simulate(self, n=100, m=8):
        state = Game.new_game()
        player = 0
        while not Game.terminal(state):
            if player == 2:
                state, r = Game.transition(state, player, 0)
                print(Game.display(state))
            else:
                _, _, _, A = MCTS(state, player, self.nnet).select_action(n, m)
                if state[player]["state"].moves[A] == 0:
                    print("invalid action selected, skipping")
                    break
                state, r = Game.transition(state, player, A)

            player = (player + 1) % 3
        return


    def train(self, replay_buffer: ReplayBuffer):
        while ray.get(replay_buffer.get_iterations.remote()) < 1000:
            time.sleep(1)

        print("starting training")
        self.nnet.train()

        i = 0
        next_batch = replay_buffer.get_batch.remote()
        while True:
            batch = ray.get(next_batch)
            next_batch = replay_buffer.get_batch.remote()
            self.train_batch(batch)
            if i % 1000 == 0:
                replay_buffer.save_state_dict.remote(deepcopy(self.nnet.state_dict()))
                path = f"checkpoints/{i:09}_checkpoint.pth"
                torch.save(self.nnet.state_dict(), path)
                print(f"Saving checkpoint to {path}")
                self.simulate()
            i += 1

    def train_batch(self, batch):
        obs, pi, v = batch
        obs = (torch.FloatTensor(i.copy()).contiguous().cuda() for i in obs)
        pi = torch.FloatTensor(pi.copy()).contiguous().cuda()
        v = torch.FloatTensor(v.copy()).contiguous().cuda()

        b1, b2, q1, q2, c1, c2 = obs

        # compute output
        with torch.autocast("cuda"):
            out_logits, out_v = self.nnet(b1, b2, q1, q2, c1, c2)
            l_pi = kl(F.log_softmax(out_logits, dim=1), F.softmax(pi, dim=1))
            l_v = mse(out_v, v)
            total_loss = l_pi + l_v

        # compute gradient and do SGD step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
