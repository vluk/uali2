import sys
from tetris.tetris_board import Board
import numpy as np
import random

random.seed(10)

# remove rng
pieces = "ioszljt" 
piece_queue = []
for i in range(500):
    new_bag = list(pieces)
    random.shuffle(new_bag)
    piece_queue += new_bag

garbage_queue = [random.randint(10) for i in range(1000)]

MIN_DELAY = 1
MAX_DELAY = 7

class PlayerState():
    def __init__(self, board=Board.new(), b2b=0, combo=-1, incoming_queue=[], piece_queue=[], t=0):
        self.board = board
        self.b2b = b2b
        self.combo = combo
        self.incoming_queue = incoming_queue
        self.piece_queue = piece_queue
        # time since last move
        self.t = t 
        self.turn_counter = 0
        self.garbage_counter = 0

        if not self.piece_queue:
            self.piece_queue = piece_queue[:4]
        
        self.moves, self.rotates = self._generate_moves()

    def _generate_moves(self):
        # instead of a hold slot, we let the player draw from either of the first two queue pieces
        # equivalent, save for end-of queue visibility at start of game
        a0, r0 = Board.generate_moves(self.board, self.piece_queue[0])
        a1, r1 = Board.generate_moves(self.board, self.piece_queue[1])
        return np.concatenate([a0, a1]), np.concatenate([r0, r1])
    
    def calculate_attack(self, lines, tspin):
        # from tetr.io combo table
        # no distinction between t-spin types
        if tspin:
            base = lines * 2
        elif lines == 4:
            base = 4
        else:
            base = lines - 1
        r = base
        if self.b2b:
            r += int(np.log(self.b2b)) + 1
        if self.combo > 0:
            r *= 1 + 0.25 * self.combo
        if base == 0 and self.combo > 1:
            r = np.log(self.combo * 1.251)
        return int(r)
    
    def terminal(self):
        return np.sum(self.board[:4]) != 0
   
    def apply_move(self, move, rng):
        # give a new playerstate and attack emitted after applying given action
        # uses supplied random number from (0,1] to deterministically generate new piece and attack columns
        i, x, y = move
        piece = i // 4
        rotate = i % 4

        piece_queue = [i for i in self.piece_queue]
        bag = [i for i in self.bag]

        board, lines = Board.apply_move(self.board, x, y, rotate, piece_queue.pop(piece))

        # use provided rng to choose which piece to add to queue
        # assumed to be unif(0, 1)
        bag_index = int(rng * len(bag))
        # re-use rng key (maybe bad)
        rng = rng * len(bag) - bag_index
        piece_queue.append(bag.pop(bag_index))
        if not bag:
            bag = list(pieces)

        combo = -1 
        b2b = self.b2b
        incoming_queue = []
        attack = 0

        if lines:
            combo = self.combo + 1
            tspin = (self.piece_queue[piece] == "t") and self.rotates[i,x,y]
            b2b = self.b2b + 1 if (tspin or (lines == 4)) else 0

            attack = self.calculate_attack(lines, tspin)
            # perfect-clear bonus
            if board.sum() == 0:
                attack += 10

            # cancel oncoming attack
            j = len(self.incoming_queue)
            while j > 0 and attack > 0:
                j -= 1
                attack -= self.incoming_queue[j]
            for i in range(j):
                incoming_queue.append(self.incoming_queue[i])
            if attack < 0:
                incoming_queue.append(-attack)
                attack = 0
        else:
            # unif (0, 1)
            for i in reversed(self.incoming_queue):
                # random int 0-9 (cryptographically secure)
                column_index = int(rng * 6)
                board = Board.apply_attack(board, i, column_index)
                # might have numerical stability issues with the rng with too much attack queue, buuut probably not
                rng = rng * 6 - column_index

        return PlayerState(board, b2b, combo, incoming_queue, piece_queue, bag, 0), lines, attack
    
    def apply_incoming(self, attack, delay):
        new_incoming = [i for i in self.incoming_queue]
        if attack:
            new_incoming.append(attack)
        return PlayerState(
            self.board,
            self.b2b,
            self.combo,
            new_incoming,
            [i for i in self.piece_queue],
            [i for i in self.bag],
            self.t + delay
        )

    def get_rep(self):
        rep = np.zeros((14, 14), dtype=int)
        # b2b indicator
        b2b_col = 0
        rep[max(rep.shape[0] - self.b2b, 0):,b2b_col] = 1 
        # combo indicator
        combo_col = 1
        rep[max(rep.shape[0] - self.combo - 1, 0):,combo_col] = 1 
        # attack indicator
        indic_col = 2
        i = rep.shape[0]
        for attack in self.incoming_queue:
            rep[max(i - attack, 0): i, indic_col] = 1 
            i -= attack + 1
        # render board
        rep[:, 3:9] = self.board
        # show queue
        queue_col = 11
        for i, piece in enumerate(reversed(self.piece_queue)):
            rep, _ = Board.apply_move(rep, i * 3 + 2, queue_col, 0, piece)
        return rep
    
    def pretty_rep(self):
        rep = self.get_rep()
 
        out = "\n".join(["".join(["██" if i else "  " for i in row]) for row in rep])
        return out

class Game():
    def new_game():
        # delay, state, action
        game = [
            {
                "state": PlayerState(),
                "delay": None,
                "move": None
            },
            {
                "state": PlayerState(),
                "delay": None,
                "move": None
            },
        ]
        return game
    
    def _apply_environment_move(game, rng):
        if game[0]["delay"] <= game[1]["delay"]:
            state_0, lines, attack = game[0]["state"].apply_move(game[0]["move"], rng)
            state_1 = game[1]["state"].apply_incoming(attack, game[0]["delay"])
            r = (lines + attack)/10
        else:
            state_1, lines, attack = game[1]["state"].apply_move(game[1]["move"], rng)
            state_0 = game[0]["state"].apply_incoming(attack, game[1]["delay"])
            r = -(lines + attack)/10
        
        return [{"state": state_0, "delay": None, "move": None}, {"state": state_1, "delay": None, "move": None}], r
    
    def _apply_player_move(game, player, delay, move):
        new_game = [game[0].copy(), game[1].copy()]
        new_game[player]["delay"] = delay
        new_game[player]["move"] = move
        if not (MIN_DELAY <= new_game[player]["state"].t + new_game[player]["delay"] <= MAX_DELAY):
            raise Exception("delay violated")
        return new_game
    
    def transition(game, player, action):
        # returns s_t+1, r(s, a)
        if player == 2:
            new_state, reward = Game._apply_environment_move(game, action)
            return (new_state, reward)
        else:
            delay, move = MIN_DELAY - game[player]["state"].t, action
            return (Game._apply_player_move(game, player, delay, move), 0)
    
    def terminal(game, player):
        return player != 2 and game[player]["state"].terminal()

    def moves(game, player):
        return game[player]["state"].moves

    def observation(game, player):
        if player == 2:
            raise Exception("environment player has no input")
        return np.concatenate([game[player]["state"].get_rep(),game[1 - player]["state"].get_rep()])
    
    def display(game):
        rep = np.concatenate([game[0]["state"].get_rep(),game[1]["state"].get_rep()], axis=1)
        return "\n".join(["".join(["  " if i else "██" for i in row]) for row in rep])

    def string_rep(rep):
        return bytearray(np.packbits(rep)).decode()
