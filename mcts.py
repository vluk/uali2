import sys
import cProfile

import numpy as np
import torch

from tetris.tetris_game import State
from tetris.nnet import NNet

c_visit = 50
c_scale = 1
discount = 0.99
epsilon = 0.0001

np.set_printoptions(threshold=sys.maxsize)

# i can't believe numpy doesn't have softmax
def softmax(x):
    r = np.exp(x - np.max(x))
    return r/r.sum()

def top_k(A, k, moves):
    """
    get top k valid moves
    """
    out = []

    indices = np.argsort(A, axis=None)
    for i in indices:
        if k == 0:
            break
        ind = np.unravel_index(i, A.shape)
        if moves[ind] == 1:
            out.append(ind)
            k -= 1

    return out

class MCTS():
    def __init__(self, state, nnet):
        self.state = state
        self.nnet = nnet

        self.moves = state.moves
        self.obs = state.obs()
        board, queue = np.expand_dims(self.obs[0], 0), np.expand_dims(self.obs[1], 0)

        logits, v = self.nnet((board, queue))
        self.logits, self.v = logits.cpu().detach().numpy()[0], v.item()
        self.g = np.random.gumbel(size=self.moves.shape)
        self.N = np.zeros_like(self.moves)
        self.r = np.zeros_like(self.moves, dtype=float)
        self.q_hat = np.zeros_like(self.moves, dtype=float)
        self.child = {}
    
    def _get_improved_estimates(self):
        """
        compute the improved policy and estimated q-value of current network using current policy, N, and q_hat
        """
        masked_pi = softmax(self.logits * (self.N != 0))

        visits = np.sum(self.N)
        if visits == 0:
            v_mix = self.v
        else:
            v_mix = (self.v + visits / np.sum(masked_pi) * np.sum(masked_pi * self.q_hat)) / (1 + visits)

        completed_q = self.q_hat + (self.N == 0) * v_mix
        new_pi = softmax(self.logits + completed_q * (c_visit + np.max(self.N)))

        # mask out invalid moves
        new_pi = new_pi * self.moves
        new_pi /= np.sum(new_pi)

        return new_pi, v_mix

    def select_action(self, n):
        """
        select action for current state using sequential halving + gumbel exploration
        """
        budget = n
        m = self.moves.sum()
        rounds = int(np.log2(m + epsilon))

        g = np.random.gumbel(size=self.logits.shape)
        A = top_k(self.logits + g, m, self.moves)

        while len(A) > 1:
            # divide budget evenly over rounds
            N_a = (n // rounds) // len(A)

            for _ in range(N_a):
                for action in A:
                    self.visit_action(action)
                    budget -= 1

            # equation (11)
            score = g + self.logits + self.q_hat * (c_visit + np.max(self.N))
            # discard lower half of scores
            A = sorted(A, key=lambda a: -score[a])[:len(A)//2]

        pi_new, v_new = self._get_improved_estimates()

        return self.obs, pi_new, v_new, A[0]

    def visit(self):
        """
        non-root exploration policy
        """
        if State.terminal(self.state):
            return 0

        new_pi, _ = self._get_improved_estimates()

        score = new_pi - self.N / (self.N.sum() + 1)
        action = np.unravel_index(np.argmax(score), score.shape)
        return self.visit_action(action)

    def visit_action(self, action):
        """
        visit a new action and update N and q_hat, creating a leaf node if necessary
        """
        if action in self.child:
            v_next = self.child[action].visit()
        else:
            next_state, r = State.transition(self.state, action)
            self.r[action] = r
            self.child[action] = MCTS(next_state, self.nnet)
            v_next = self.child[action].v
        
        q = self.r[action] + v_next * discount

        q_sum = self.q_hat[action] * self.N[action] + q
        self.N[action] += 1
        self.q_hat[action] = q_sum / self.N[action]
        return q

nnet = NNet().cuda()
state = State.new()
cProfile.run('MCTS(state, nnet).select_action(500)')