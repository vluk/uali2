import numpy as np
import random

from tetris.tetris_board import Board, minos

MAX_GARBAGE = 9
MAX_QUEUE = 7

mapping = {
    "i" : 0,
    "o" : 1,
    "s" : 2,
    "z" : 3,
    "l" : 4,
    "j" : 5,
    "t" : 6
}

class State():
    def __init__(self, board, garbage, queue, bag, rng: random.Random):
        self.board = board
        self.garbage = garbage
        self.queue = queue
        self.bag = bag
        self.rng = rng
        self.moves = self._generate_moves()

    def _generate_moves(self):
        # instead of a hold slot, we let the player draw from either of the first two queue pieces
        # equivalent, save for end-of queue visibility at start of game
        a0 = Board.generate_moves(self.board, self.queue[0])
        a1 = Board.generate_moves(self.board, self.queue[1])
        return np.concatenate([a0, a1])
    
    @staticmethod
    def new():
        rng = random.Random()
        garbage = 16
        queue = minos.copy()
        bag = minos.copy()

        rng.shuffle(queue)
        rng.shuffle(bag)
        cheese = []
        for _ in range(garbage):
            cheese.append(rng.randrange(10))

        board = Board.apply_garbage(Board.new(), cheese)

        return State(board, garbage, queue, bag, rng)

    def terminal(self):
        return (np.sum(self.moves) == 0) or (self.garbage == 0)
   
    def transition(self, action):
        i, x, y = action
        piece = i // 4
        rotate = i % 4

        queue = self.queue.copy()
        bag = self.bag.copy()

        board, cleared = Board.apply_move(self.board, x, y, rotate, queue.pop(piece))
        queue.append(bag.pop())

        if len(bag) == 0:
            bag = minos.copy()
            self.rng.shuffle(bag)

        cleared = cleared[-self.garbage:].sum()
        garbage = self.garbage - cleared

        return State(board, garbage, queue, bag, random.Random(self.rng.getstate)), cleared
    
    def obs(self):
        b = self.board[4:]
        q = np.zeros((7, 7))

        for i, piece in enumerate(self.queue):
            q[i][mapping[piece]] = 1

        return (b, q)
    
    def print(self):
        out = ""
        x, y = self.board.shape
        for i in range(x):
            for j in range(y):
                if self.board[i][j] == 0:
                    out += "  "
                else:
                    out += "██"
            out += "\n"
        print(out)
        print(self.queue)
