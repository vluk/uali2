import ray

from tetris.tetris_game import State
from tetris.nnet import NNet
from mcts import MCTS

discount = 0.99

class Game():
    def __init__(self, td_steps=5):
        self.td_steps = td_steps

        self.steps = 0
        self.observations = []
        self.policies = []
        self.values = []
        self.rewards = []
        self.value_targets = []
    
    def add(self, obs, pi, v, r):
        self.steps += 1
        self.observations.append(obs)
        self.policies.append(pi)
        self.values.append(v)
        self.rewards.append(r)
    
    def compute_value_targets(self):
        values = self.values + [0] * self.td_steps

        for step in range(self.steps):
            bootstrap = step + self.td_steps
            if bootstrap < self.steps:
                value = values[bootstrap] * discount**self.td_steps
            else:
                value = 0
            
            for i, reward in enumerate(self.rewards[step+1:bootstrap]):
                value += reward * discount**i
            self.value_targets.append(value)

    def save(self, replay_buffer):
        self.compute_value_targets()

        ep_reward = 0
        t = 1
        
        for step in range(self.steps):
            target = (
                self.observations[step],
                self.policies[step],
                self.value_targets[step],
                self.rewards[step]
            )
            ep_reward += self.rewards[step] * t
            t *= discount
            replay_buffer.save_target.remote(target)

        print(f"Episode reward:{ep_reward}")

@ray.remote(num_gpus=0.1)
class SelfPlay():
    def __init__(self, initial_state_dict):
        self.steps = 0
        self.reward = 0
        self.i = 0
        self.nnet = NNet().cuda()
        self.nnet.load_state_dict(initial_state_dict)

    def run_game(self, replay_buffer, n=500):
        state = State.new()
        game = Game()

        while not state.terminal():
            obs, pi, v, a = MCTS(state, self.nnet).select_action(n)
            state, r = state.transition(a)
            game.add(obs, pi, v, r)
        
        game.save(replay_buffer)

    def get_latest_game(self):
        return self.steps, self.reward, self.i
    
    def selfplay(self, replay_buffer):
        i = 0
        while True:
            self.run_game(replay_buffer)

            self.nnet.load_state_dict(ray.get(replay_buffer.get_state_dict.remote()))
            self.nnet.eval()
            i += 1