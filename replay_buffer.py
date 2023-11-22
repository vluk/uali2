import numpy as np
import ray

discount = 0.95
td_steps = 5

@ray.remote(num_gpus=0.05)
class ReplayBuffer():
    def __init__(self, initial_state_dict, window_size=100000, batch_size=256, unroll_steps=5):
        self.window_size = window_size
        self.batch_size = batch_size
        self.unroll_steps = unroll_steps
        self.obs_buffer = [
            np.zeros((window_size, 20, 10)),
            np.zeros((window_size, 7, 7)),
        ]
        self.pi_buffer = np.zeros((window_size, 8, 20, 10))
        self.v_buffer = np.zeros((window_size, 1))
        self.r_buffer = np.zeros((window_size, 1))
        self.i = 0
        self.full = False
        self.state_dict = initial_state_dict

    def get_iterations(self):
        return self.i

    def get_state_dict(self):
        return self.state_dict

    def save_state_dict(self, state_dict):
        self.state_dict = state_dict
    
    def save_target(self, target):
        obs, pi, v, r = target

        self.obs_buffer[0][self.i] = obs[0]
        self.obs_buffer[1][self.i] = obs[1]

        self.pi_buffer[self.i] = pi
        self.v_buffer[self.i] = v
        self.r_buffer[self.i] = r
        
        self.i += 1

        if self.i == self.window_size:
            self.i = 0
            self.full = True

    def get_batch(self): 
        indices = np.random.choice(np.arange(self.window_size if self.full else self.i), self.batch_size)
        obs = []
        for i in range(2):
            obs.append(self.obs_buffer[i][indices])
        return (obs, self.pi_buffer[indices], self.v_buffer[indices])
