import ray

from tetris.nnet import NNet

from replay_buffer import ReplayBuffer
from self_play import SelfPlay
from trainer import Trainer

def main(num_workers = 8):
    nnet = NNet()
    initial_state_dict = nnet.state_dict()

    replay_buffer = ReplayBuffer.remote(initial_state_dict)
    workers = [SelfPlay.remote(initial_state_dict) for _ in range(num_workers)]
    trainer = Trainer.remote(initial_state_dict)

    [worker.selfplay.remote(replay_buffer) for worker in workers]
    ray.get(trainer.train.remote(replay_buffer))

if __name__ == "__main__":
    ray.init(num_gpus=1, dashboard_port=8888)
    main()