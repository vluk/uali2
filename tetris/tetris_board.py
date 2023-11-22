import numpy as np
from numpy import clip, copy
from scipy.signal import correlate2d
import math
import random


"""
After some thought, an input-based move generation approach has some
fundamental performance issues. We can use numpy to generate moves at
a much higher speed.
"""

"""from tetrio.js"""

kicks = {
    1: [[0, 0], [-1, 0], [-1, -1], [0, 2], [-1, 2]],
    10: [[0, 0], [1, 0], [1, 1], [0, -2], [1, -2]],
    12: [[0, 0], [1, 0], [1, 1], [0, -2], [1, -2]],
    21: [[0, 0], [-1, 0], [-1, -1], [0, 2], [-1, 2]],
    23: [[0, 0], [1, 0], [1, -1], [0, 2], [1, 2]],
    32: [[0, 0], [-1, 0], [-1, 1], [0, -2], [-1, -2]],
    30: [[0, 0], [-1, 0], [-1, 1], [0, -2], [-1, -2]],
    3: [[0, 0], [1, 0], [1, -1], [0, 2], [1, 2]],
    2: [[0, 0], [0, -1], [1, -1], [-1, -1], [1, 0], [-1, 0]],
    13: [[0, 0], [1, 0], [1, -2], [1, -1], [0, -2], [0, -1]],
    20: [[0, 0], [0, 1], [-1, 1], [1, 1], [-1, 0], [1, 0]],
    31: [[0, 0], [-1, 0], [-1, -2], [-1, -1], [0, -2], [0, -1]]
}

"""converted from tetrio i kick table using TTC implementation offsets"""
"""read: https://harddrop.com/wiki/SRS#How_Guideline_SRS_Really_Works"""

i_kicks = {
    1: [[1, 0], [2, 0], [-1, 0], [-1, 1], [2, -2]],
    10: [[-1, 0], [-2, 0], [1, 0], [-2, 2], [1, -1]],
    12: [[0, -1], [-1, -1], [2, -1], [-1, -3], [2, 0]],
    21: [[0, 1], [-2, 1], [1, 1], [-2, 0], [1, 3]],
    23: [[-1, 0], [1, 0], [-2, 0], [1, -1], [-2, 2]],
    32: [[1, 0], [2, 0], [-1, 0], [2, -2], [-1, 1]],
    30: [[0, 1], [1, 1], [-2, 1], [1, 3], [-2, 0]],
    3: [[0, -1], [-1, -1], [2, -1], [2, 0], [-1, -3]],
    2: [[1, -1], [1, -2]],
    13: [[-1, -1], [0, -1]],
    20: [[-1, 1], [-1, 2]],
    31: [[1, 1], [0, 1]]
}

o_kicks = {
    1: [[0, 1]],
    2: [[1, 1]],
    3: [[1, 0]],
    10: [[0, -1]],
    12: [[1, 0]],
    13: [[1, -1]],
    20: [[-1, -1]],
    21: [[-1, 0]],
    23: [[0, -1]],
    30: [[-1, 0]],
    31: [[-1, 1]],
    32: [[0, 1]]
}

kicks = {i: [(max(x, 0), max(y, 0), max(-x, 0), max(-y, 0)) for y,x in kicks[i]] for i in kicks}
o_kicks = {i: [(max(x, 0), max(y, 0), max(-x, 0), max(-y, 0)) for y,x in o_kicks[i]] for i in o_kicks}
i_kicks = {i: [(max(x, 0), max(y, 0), max(-x, 0), max(-y, 0)) for y,x in i_kicks[i]] for i in i_kicks}

TRIL = np.tril(np.ones((24, 24), dtype=int), k=0)
POW = np.power(2, np.arange(24))


minos = ["z", "l", "o", "s", "i", "j", "t"]

pieces = {
    "i": np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=np.int64),
    "j": np.array([[1, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.int64),
    "l": np.array([[0, 0, 1], [1, 1, 1], [0, 0, 0]], dtype=np.int64),
    "o": np.array([[0, 1, 1], [0, 1, 1], [0, 0, 0]], dtype=np.int64),
    "s": np.array([[0, 1, 1], [1, 1, 0], [0, 0, 0]], dtype=np.int64),
    "t": np.array([[0, 1, 0], [1, 1, 1], [0, 0, 0]], dtype=np.int64),
    "z": np.array([[1, 1, 0], [0, 1, 1], [0, 0, 0]], dtype=np.int64)
}

rots = {mino: [np.rot90(pieces[mino], i, axes=(1, 0))
               for i in range(4)] for mino in minos}

class Board():
    @staticmethod
    def _drop(moves, full):
        """simulate soft drop"""
        moves -= full
        # TRI "drips" downwards
        # POW gives "priority" to lower elements
        moves = moves * POW[:, np.newaxis]
        moves = TRIL @ moves
        moves = clip(moves, 0, 1)
        moves[:,:23] -= moves[:,1:]
        moves = clip(moves, 0, 1)
        return moves

    @staticmethod
    def _rotate(moves, full, piece):
        h = moves.shape[1]
        w = moves.shape[2]
        rotated = np.zeros_like(moves)
        kick = o_kicks if piece == "o" else (i_kicks if piece == "i" else kicks)
        # loop through source and target rotations
        for i in kicks:
            mask = 1 - full[i % 10]
            move = copy(moves[i // 10])
            for x, y, nx, ny in kick[i]:
                # add with offset, then subtract from source
                valids = move[nx:h-x, ny:w-y] & mask[x:h-nx, y:w-ny]
                moves[i % 10][x:h-nx, y:w-ny] += valids
                rotated[i % 10][x:h-nx, y:w-ny] += valids
                move[nx:h-x, ny:w-y] -= valids

        moves = clip(moves, 0, 1)
        return moves, rotated

    @staticmethod
    def generate_moves(board, piece):
        # implementation of srs+
        # at most 3 rotates
        # doesn't calculate tucks
        # we assume there's no top-of-board collisions
        board = board.copy()
        h = board.shape[0]
        w = board.shape[1]
        moves = np.zeros((4, h, w), dtype=int)

        if board[:4].sum() > 0:
            return moves[:, 4:]

        # padding of piece
        x = rots[piece][0].shape[0] // 2
        # general convolution, not performant
        full = np.array([correlate2d(board, rots[piece][i], fillvalue=1)[x:h+x,x:w+x] for i in range(4)], dtype=int)
        full = clip(full, 0, 1)
        moves[:, 3] = 1
        moves[:, 3] -= full[:, 3]
        moves = Board._drop(moves, full)
        for _ in range(3):
            moves, r = Board._rotate(moves, full, piece)
            moves = Board._drop(moves, full)
        return moves[:, 4:]

    @staticmethod
    def apply_move(board, x, y, r, p):
        # add piece to board, clearing lines and returning number of cleared lines
        # also returns list of cleared lines
        x += 4
        board = board.copy()
        piece = rots[p][r]

        w = 5 if p == "i" else 3
        pad = w // 2
        for i in range(w):
            for j in range(w):
                if piece[i][j]:
                    board[x + i - pad, y + j - pad] = 1

        cleared = np.sum(board, axis=1) == board.shape[1]
        board = np.concatenate([np.zeros((cleared.sum(), board.shape[1]), dtype=int), board[~cleared]])

        return board, cleared

    @staticmethod
    def apply_garbage(board, garbage):
        # pushes board up with garbage in attack column
        # clips beyond line 24
        board = board.copy()
        h = board.shape[0]
        for column in garbage:
            board[:h - 1] = board[1:]
            board[h - 1:h] = 1
            board[h - 1:h, column] = 0
        return board
    
    @staticmethod
    def new(h = 24, w = 10):
        return np.zeros((h, w), dtype=int)
