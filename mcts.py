import numpy as np
from tetris.tetris_game import Game
import torch
import random

c_visit = 50
c_scale = 1
discount = 0.99

# i can't believe numpy doesn't have softmax
def softmax(x):
    r=np.exp(x - np.max(x))
    return r/r.sum()

def top_k(A, k):
    return [np.unravel_index(i, A.shape) for i in np.argpartition(A.flatten(), -k)[-k:]]

class MCTS():
    def __init__(self, state, player, nnet):
        self.state = state
        self.player = player
        self.nnet = nnet

        self.is_chance = self.player == 2

        if self.is_chance:
            self.rng = random.random()
            _, self.v = self.nnet.predict(Game.observation(self.state, 0))
        else:
            self.moves = Game.moves(self.state, self.player)
            self.obs = Game.observation(state, player)

            self.logits, self.v = self.nnet.predict(self.obs)
            self.g = np.random.gumbel(size=self.moves.shape)
            self.N = np.zeros_like(self.moves)
            self.q_hat = np.zeros_like(self.moves)

        self.child = {}
    
    def _get_improved_estimates(self):
        masked_pi = softmax(self.logits) * (self.N != 0)

        visits = np.sum(self.N)
        if visits == 0:
            v_mix = self.v
        else:
            v_mix = (self.v + visits / np.sum(masked_pi) * np.sum(masked_pi * self.q_hat)) / (1 + visits)

        completed_q = self.q_hat + (self.N == 0) * v_mix
        new_pi = softmax(self.logits + completed_q * (c_visit + np.max(self.N)))

        return new_pi, v_mix

    def select_action(self, n=200, m=8):
        if self.player == 2:
            return None, None, None, self.rng
        rounds = int(np.log2(m))
        g = np.random.gumbel(size=self.logits.shape)
        remainder = n
        while m > 1:
            # equation 8
            budget = n // rounds + (remainder if m == 2 else 0)
            N_a = budget // m
            remainder -= N_a * m

            score = g + self.logits + self.q_hat * (c_visit + np.max(self.N))
            A = top_k(np.argsort(score), m)

            for _ in range(N_a):
                for action in A:
                    self.visit_action(action)

            m //= 2

        score = g + self.logits + self.q_hat * (c_visit + np.max(self.N))
        A_new = np.unravel_index(np.argmax(score * self.moves), score.shape)

        pi_new, v_new = self._get_improved_estimates()

        return self.obs, pi_new, v_new, A_new

    def visit(self):
        if self.is_chance:
            return self.visit_action(self.rng)
        if Game.terminal(self.state, self.player):
            return -1 

        new_pi, _ = self._get_improved_estimates()

        score = new_pi - self.N / (self.N.sum() + 1)
        action = np.unravel_index(np.argmax(score * self.moves), score.shape)
        return self.visit_action(action)

    def visit_action(self, action):
        # invalid move loses game
        if not self.player == 2 and self.moves[action] == 0:
            self.N[action] += 1
            self.q_hat[action] -= 1
            return -1
        if str(action) in self.child:
            # zero sum game
            q = discount * -self.child[str(action)].visit()
        else:
            next_state, r = Game.transition(self.state, self.player, action)
            self.child[str(action)] = MCTS(next_state, (self.player + 1) % 3, self.nnet)
            q = -(discount * self.child[str(action)].v + r)
        
        if self.is_chance:
            # perspective of chance node is player 0
            # not really sure this is the best way to express it
            # the minus sign in the prior step should really be assigned to the next_value
            return -q
        else:
            q_sum = self.q_hat[action] * self.N[action] + q
            self.N[action] += 1
            self.q_hat[action] = q_sum / self.N[action]
            return q