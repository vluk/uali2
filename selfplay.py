import torch
import numpy as np
import cProfile

from tetris.tetris_game import Game
from mcts import MCTS
from tetris.nnet import NNetWrapper

class ReplayBuffer():
    def __init__(self, window_size=10000, batch_size=16):
        self.window_size = window_size
        self.batch_size = batch_size
        self.obs_buffer = [
            np.zeros((window_size, 10, 6)),
            np.zeros((window_size, 10, 6)),
            np.zeros((window_size, 7, 7)),
            np.zeros((window_size, 7, 7)),
            np.zeros((window_size, 3, 20)),
            np.zeros((window_size, 3, 20))
        ]
        self.pi_buffer = np.zeros((window_size, 8 * 10 * 6))
        self.v_buffer = np.zeros((window_size, 1))
        self.i = 0
        self.full = False
    
    def save_experience(self, experience):
        obs, pi, v = experience
        for j in range(6):
            self.obs_buffer[j][self.i] = obs[j]
        self.pi_buffer[self.i] = pi.flatten()
        self.v_buffer[self.i] = v
        self.i += 1
        if self.i == self.window_size:
            self.i = 0
            self.full = True

    def get_batch(self): 
        indices = np.random.choice(np.arange(self.window_size if self.full else self.i), self.batch_size)
        obs = (self.obs_buffer[j][indices] for j in range(6))
        return obs, self.pi_buffer[indices], self.v_buffer[indices]

def run_selfplay(nnet, replay_buffer, n=200, m=16, display = False):
    state = Game.new_game()
    player = 0
    steps = 0
    reward_encountered = 0 
    while not Game.terminal(state):
        if player == 0 and display:
            print(Game.display(state))

        if player == 2:
            state, r = Game.transition(state, player, 0)
        else:
            obs, pi, v, A = MCTS(state, player, nnet).select_action(n, m)
            if state[player]["state"].moves[A] == 0:
                print("invalid action selected, skipping")
                break
            state, r = Game.transition(state, player, A)
            replay_buffer.save_experience((obs, pi, v))

        player = (player + 1) % 3

        reward_encountered += abs(r)
        steps += 1
    return steps, reward_encountered


def main(iterations, games):
    replay_buffer = ReplayBuffer()
    nnet = NNetWrapper()
    for i in range(iterations):
        print(f"Starting iteration {i}...")
        for j in range(games):
            steps, reward_encountered = run_selfplay(nnet, replay_buffer)
            print(f"Game {j}: {steps} steps, {reward_encountered} total reward")
#         nnet.train(replay_buffer)
#         if i % 10 == 0:
#             fname = f"checkpoints/{i:04}_checkpoint.pth"
#             print(f"Saving checkpoint at {fname}...")
#             torch.save(nnet.nnet, fname)
#             run_selfplay(nnet, replay_buffer, n=200, m=16, display=True)


if __name__ == "__main__":
    cProfile.run("main(1, 1)")
