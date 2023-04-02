import time
from copy import deepcopy

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
import ray

from tetris.timing_nnet import TimingNNet
from mcts import MCTS
from tetris.tetris_game import Game, MIN_DELAY, MAX_DELAY
from replay_buffer import ReplayBuffer

kl = nn.KLDivLoss()
mse = nn.MSELoss()

eps = 0.0001

@ray.remote(num_gpus=0.15)
class Trainer():
    def __init__(self, initial_state_dict):
        self.nnet = TimingNNet().cuda()
        self.nnet.load_state_dict(initial_state_dict)
        self.optimizer = optim.Adam(self.nnet.parameters(), lr=0.00005)

    def simulate(self, n=200, m=16):
        state = Game.new_game()

        while not Game.terminal(state):
            print(Game.display(state))
            # both players make moves simultaneously based on observation
            obs, pi, v, A0, d0  = MCTS(Game.view(state, 0), 0, self.nnet).select_action(n, m)
            if state[0]["state"].moves[A0] == 0:
                print("invalid action selected, skipping")

            obs, pi, v, A1, d1 = MCTS(Game.view(state, 1), 0, self.nnet).select_action(n, m)
            if state[1]["state"].moves[A1] == 0:
                print("invalid action selected, skipping")

            state, _ = Game.transition(state, 0, A0, d0)
            state, _ = Game.transition(state, 1, A1, d1)

            state, _ = Game.transition(state, 2, 0, 0)

        return

    def train(self, replay_buffer: ReplayBuffer):
        while ray.get(replay_buffer.get_iterations.remote()) < 1000:
            time.sleep(1)

        print("starting training")
        self.nnet.train()

        i = 87000 
        next_batch = replay_buffer.get_batch.remote()
        while True:
            batch = ray.get(next_batch)
            next_batch = replay_buffer.get_batch.remote()
            self.train_batch_timing(batch)
            if i % 1000 == 0:
                replay_buffer.save_state_dict.remote(deepcopy(self.nnet.state_dict()))
                path = f"checkpoints2/{i:09}_checkpoint.pth"
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

    
    def train_batch_timing(self, batch):
        self.nnet.train()
        obs, pi, v = batch
        obs = (torch.FloatTensor(i.copy()).contiguous().cuda() for i in obs)
        pi = torch.FloatTensor(pi.copy()).contiguous().cuda()
        v = torch.FloatTensor(v.copy()).contiguous().cuda()

        b1, b2, q1, q2, c1, c2 = obs

        self.optimizer.zero_grad()

        # compute output
        out_logits, out_v, out_d = self.nnet(b1, b2, q1, q2, c1, c2)

        # get gradient of delay
        # setting it regularly doesn't work, so workaround
        g = []

        def hook(grad):
            g.append(grad)

        out_d.register_hook(hook)
        out_v.backward(torch.ones_like(out_v), retain_graph=True)

        # inverting gradients
        t = c1[2]
        d_max = MAX_DELAY - t
        d_min = torch.clamp(MIN_DELAY - t, 0)
        d_grad = eps * g[0] * (d_max - d_min + torch.sign(g[0]) * (d_max + d_min - 2 * out_d))

        self.optimizer.zero_grad()

        # get gradient of delay
        l_pi = kl(F.log_softmax(out_logits, dim=1), pi)
        l_v = mse(out_v, v)
        total_loss = l_pi + l_v
        if out_logits.sum() > 1000:
            print(F.softmax(out_logits, dim=1))
            print(pi)
            print(total_loss)

        total_loss.backward(retain_graph=True)
        # induce gradient on timing variable
        out_d.backward(d_grad)

        self.optimizer.step()