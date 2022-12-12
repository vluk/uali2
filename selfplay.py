import torch
import numpy as np

from tetris.tetris_game import Game
from mcts import MCTS
from tetris.nnet import NNetWrapper

class ReplayBuffer():
    def __init__(self, window_size=100000, batch_size=16):
        self.window_size = window_size
        self.batch_size = batch_size
        self.obs_buffer = np.zeros((window_size, 28, 14))
        self.pi_buffer = np.zeros((window_size, 8 * 14 * 6))
        self.v_buffer = np.zeros((window_size, 1))
        self.i = 0
        self.full = False
    
    def save_experience(self, experience):
        obs, pi, v = experience
        self.obs_buffer[self.i] = obs
        self.pi_buffer[self.i] = pi.flatten()
        self.v_buffer[self.i] = v
        self.i += 1
        if self.i == self.window_size:
            self.i = 0
            self.full = True

    def get_batch(self): 
        indices = np.random.choice(np.arange(self.window_size if self.full else self.i), self.batch_size)
        return self.obs_buffer[indices], self.pi_buffer[indices], self.v_buffer[indices]

def run_selfplay(nnet, replay_buffer, display = False):
    state = Game.new_game()
    player = 0
    steps = 0
    reward_encountered = 0 
    while not Game.terminal(state, player):
        if player == 0 and display:
            print(Game.display(state))

        obs, pi, v, A = MCTS(state, player, nnet).select_action()
        if player != 2:
            replay_buffer.save_experience((obs, pi, v))

        state, r = Game.transition(state, player, A)
        player = (player + 1) % 3

        reward_encountered += abs(r)
        steps += 1
    return steps, reward_encountered


def main():
    replay_buffer = ReplayBuffer()
    nnet = NNetWrapper()
    for i in range(500):
        print(f"Starting iteration {i}...")
        for j in range(25):
            steps, reward_encountered = run_selfplay(nnet, replay_buffer)
            print(f"Game {j}: {steps} steps, {reward_encountered} total reward")
        nnet.train(replay_buffer)
        run_selfplay(nnet, replay_buffer, display=True)
    

if __name__ == "__main__":
    main()
