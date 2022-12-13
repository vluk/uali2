import torch
import numpy as np
import ray

from tetris.tetris_game import Game
from mcts import MCTS
from tetris.nnet import NNet

@ray.remote(num_gpus=0.1)
class SelfPlay():
    def __init__(self, initial_state_dict):
        self.nnet = NNet()
        self.nnet.load_state_dict(initial_state_dict)
        self.nnet.cuda()
        self.nnet.eval()
        self.steps = 0
        self.reward = 0
        self.i = 0

    def run_game(self, replay_buffer, n=200, m=16):
        state = Game.new_game()
        player = 0
        steps = 0
        reward_encountered = 0
        while not Game.terminal(state):
            if player == 2:
                state, r = Game.transition(state, player, 0)
            else:
                obs, pi, v, A = MCTS(state, player, self.nnet).select_action(n, m)
                if state[player]["state"].moves[A] == 0:
                    print("invalid action selected, skipping")
                    break
                state, r = Game.transition(state, player, A)
                replay_buffer.save_experience.remote((obs, pi, v))

            player = (player + 1) % 3

            reward_encountered += abs(r)
            steps += 1
        return steps, reward_encountered
    
    def get_latest_game(self):
        return self.steps, self.reward, self.i
    
    def selfplay(self, replay_buffer):
        i = 0
        while True:
            steps, reward = self.run_game(replay_buffer)
            print(f"Game {i}: {steps} steps, {reward} total reward")

            self.steps, self.reward, self.i = steps, reward, i

            if i % 25 == 0:
                self.nnet.load_state_dict(ray.get(replay_buffer.get_state_dict.remote()))
                self.nnet.cuda()
                self.nnet.eval()
            i += 1
