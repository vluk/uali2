import ray
import random
import time
import torch

from tetris.timing_nnet import TimingNNet
from tetris.nnet import NNet

from replay_buffer import ReplayBuffer
from self_play import SelfPlay
from trainer import Trainer

def main(num_workers = 8):
    nnet = TimingNNet()
    nnet.load_state_dict(torch.load("checkpoints2/000087000_checkpoint.pth"))
    initial_state_dict = nnet.state_dict()

    replay_buffer = ReplayBuffer.remote(initial_state_dict)
    workers = [SelfPlay.remote(initial_state_dict) for _ in range(num_workers)]
    trainer = Trainer.remote(initial_state_dict)

    [worker.selfplay.remote(replay_buffer) for worker in workers]
    ray.get(trainer.train.remote(replay_buffer))

def evaluate(num_workers = 10):
    workers = [SelfPlay.remote() for _ in range(num_workers)]

    w = [worker.evaluate.remote() for worker in workers]
    ray.get(w[0])

if __name__ == "__main__":
    ray.init(num_gpus=1, dashboard_port=8888)
    evaluate()
    ray.shutdown()