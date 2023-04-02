import numpy as np
import ray

@ray.remote(num_gpus=0.05)
class ReplayBuffer():
    def __init__(self, initial_state_dict, window_size=100000, batch_size=16):
        self.window_size = window_size
        self.batch_size = batch_size
        self.obs_buffer = [
            np.zeros((window_size, 10, 6)),
            np.zeros((window_size, 10, 6)),
            np.zeros((window_size, 7, 7)),
            np.zeros((window_size, 7, 7)),
            np.zeros((window_size, 4)),
            np.zeros((window_size, 4))
        ]
        self.pi_buffer = np.zeros((window_size, 8 * 10 * 6))
        self.v_buffer = np.zeros((window_size, 1))
        self.i = 0
        self.full = False
        self.state_dict = initial_state_dict
    
    def get_iterations(self):
        return self.i
    
    def get_state_dict(self):
        return self.state_dict
    
    def save_state_dict(self, state_dict):
        self.state_dict = state_dict
    
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
        obs = []
        for j in range(6):
            obs.append(self.obs_buffer[j][indices])
        return (obs, self.pi_buffer[indices], self.v_buffer[indices])
