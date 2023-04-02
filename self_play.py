import torch
import numpy as np
import ray

from tetris.tetris_game import Game, MIN_DELAY, MAX_DELAY
from mcts import MCTS
from tetris.timing_nnet import TimingNNet

@ray.remote(num_gpus=0.1)
class SelfPlay():
    def __init__(self):
        self.steps = 0
        self.reward = 0
        self.i = 0

    def run_game(self, replay_buffer, n=200, m=8):
        state = Game.new_game()
        steps = 0
        reward_encountered = 0

        while not Game.terminal(state):
            # both players make moves simultaneously based on observation
            obs, pi, v, A0, d0  = MCTS(Game.view(state, 0), 0, self.nnet).select_action(n, m)
            if state[0]["state"].moves[A0] == 0:
                print("invalid action selected, skipping")
                break
            replay_buffer.save_experience.remote((obs, pi, v))

            obs, pi, v, A1, d1 = MCTS(Game.view(state, 1), 0, self.nnet).select_action(n, m)
            if state[1]["state"].moves[A1] == 0:
                print("invalid action selected, skipping")
                break
            replay_buffer.save_experience.remote((obs, pi, v))

            state, _ = Game.transition(state, 0, A0, d0)
            state, _ = Game.transition(state, 1, A1, d1)

            state, r = Game.transition(state, 2, 0, 0)

            reward_encountered += abs(r)
            steps += 1

        return steps, reward_encountered

    def evaluate(self):
        nnet = TimingNNet()
        for i in range(50):
            for j in range(2):
                state = Game.new_game()
                nnet.load_state_dict(torch.load(f"checkpoints2/{i*2000:09}_checkpoint.pth"))
                nnet.cuda()
                nnet.eval()
                while not Game.terminal(state):
                    _, _, _, A0, d0 = MCTS(Game.view(state, 0), 0, nnet).select_action(100, 8)
                    _, _, _, A1, _ = MCTS(Game.view(state, 1), 0, nnet).select_action(100, 8)
                    if state[0]["state"].moves[A0] == 0:
                        print(i, 1)
                        break
                    if state[1]["state"].moves[A1] == 0:
                        print(i, 0)
                        break
                    state, _ = Game.transition(state, 0, A0, d0)
                    state, _ = Game.transition(state, 1, A1, MIN_DELAY - state[1]["state"].t + i / 100)

                    state, r = Game.transition(state, 2, 0, 0)
                    if (state[0]["state"].terminal()):
                        print(i, 1)
                    if (state[1]["state"].terminal()):
                        print(i, 0)


    def get_latest_game(self):
        return self.steps, self.reward, self.i
    
    def selfplay(self, replay_buffer):
        i = 0
        while True:
            steps, reward = self.run_game(replay_buffer)
            print(f"Game {i}: {steps} steps, {reward} total reward")

            self.steps, self.reward, self.i = steps, reward, i

            self.nnet.load_state_dict(ray.get(replay_buffer.get_state_dict.remote()))
            self.nnet.eval()
            i += 1
