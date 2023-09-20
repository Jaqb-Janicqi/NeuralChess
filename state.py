from copy import deepcopy

import numpy as np

from move import Move

npstate = np.dtype([
    ("-1", np.uint64),
    ("1", np.uint64),
    ("-2", np.uint64),
    ("2", np.uint64),
    ("-3", np.uint64),
    ("3", np.uint64),
    ("-4", np.uint64),
    ("4", np.uint64),
    ("-5", np.uint64),
    ("5", np.uint64),
    ("-6", np.uint64),
    ("6", np.uint64),
    ("castle_rights", np.uint64),
    ("en_passant_target", np.uint64),
    ("turn", np.int8),
    ("winner", np.uint8),
    ("terminal", np.bool_),
    ("halfmove", np.uint8),
    ("fullmove", np.uint8),
    ("fiftymoverule", np.uint8)
])


class State():
    def __init__(self) -> None:
        self.__game_state: np.ndarray = np.zeros((), dtype=npstate)
        self.__board = np.zeros((8, 8), dtype=np.int8)

    def __getitem__(self, key) -> np.ndarray:
        return self.__game_state[key].item()

    def __setitem__(self, key, value) -> None:
        self.__game_state[key] = value

    def __str__(self) -> str:
        return str(self.__game_state)

    def copy(self):
        return deepcopy(self.__game_state)

    def encode(self):
        board = self.__board if self.__game_state["turn"] == 1 else np.flip(
            -self.__board, axis=(0, 1))
        stack = np.stack((
            np.where(board > 0, board, 0),
            np.where(board == 0, 1, 0),
            np.where(board < 0, board, 0))
        ).astype(np.float32)
        return stack

    def pieces_of(self, player: int) -> np.uint64:
        if player == 1:
            return self.white
        return self.black

    def get_next_state(self, move: Move):
        next_state = self.copy()
        next_state.make_move(move)
        return next_state

    def make_move(self, move: Move) -> None:
        src_piece = self.__board[move.src_y, move.src_x].item()
        dst_piece = self.__board[move.dst_y, move.dst_x].item()

        # check if castle and move rook
        if move.castle != 0:
            if move.castle == 1:
                # move rook on the board
                self.__board[move.dst_y, move.dst_x +
                             1] = self.__board[move.dst_y, 7]
                self.__board[move.dst_y, 7] = 0
                # move rook on the bitboard
                self.__game_state[4*self.player_turn] /= np.uint64(
                    1 << (move.dst_y * 8 + move.dst_x + 1))
                self.__game_state[4*self.player_turn] ^= np.uint64(
                    1 << (move.dst_y * 8 + 7))
            else:
                # move rook on the board
                self.__board[move.dst_y, move.dst_x -
                             1] = self.__board[move.dst_y, 0]
                self.__board[move.dst_y, 0] = 0
                # move rook on the bitboard
                self.__game_state[4*self.player_turn] /= np.uint64(
                    1 << (move.dst_y * 8 + move.dst_x - 1))
                self.__game_state[4*self.player_turn] ^= np.uint64(
                    1 << (move.dst_y * 8 + 0))
            # update castle rights
            if self.player_turn == 1:
                # remove white castle rights
                self.__game_state["castle_rights"] ^= np.uint64(0b11)
            else:
                # remove black castle rights
                self.__game_state["castle_rights"] ^= np.uint64(0b1100)
                
        # check if en passant and remove captured pawn from board and bitboard
        elif move.src_y * 8 + move.src_x == self.en_passant_target:
            self.__board[move.src_y - self.player_turn, move.dst_x] = 0
            self.__game_state[-self.player_turn] ^= np.uint64(
                1 << (move.src_y - self.player_turn) * 8 + move.dst_x)  
        # check if promotion and add new piece
        elif move.promo_piece != 0:
            self.__game_state[move.promo_piece] /= np.uint64(
                1 << (move.dst_y * 8 + move.dst_x))
            self.__game_state[src_piece] ^= np.uint64(
                1 << (move.src_y * 8 + move.src_x))
            self.__board[move.dst_y, move.dst_x] = move.promo_piece

        self.__board[move.src_y, move.src_x] = 0
        self.__board[move.dst_y, move.dst_x] = src_piece

        # update bitboards
        src_bb = np.uint64(1 << (move.src_y * 8 + move.src_x))
        dst_bb = np.uint64(1 << (move.dst_y * 8 + move.dst_x))
        self.__game_state[src_piece] ^= src_bb
        self.__game_state[dst_piece] ^= dst_bb

        # update en passant target if a pawn moved 2 squares
        if src_piece == 1 and abs(move.src_y - move.dst_y) == 2:
            self.__game_state["en_passant_target"] = move.src_y * 8 + move.src_x
        else:
            self.__game_state["en_passant_target"] = 0

        # update game state
        self.__game_state["turn"] = -self.__game_state["turn"]
        self.__game_state["halfmove"] += 1
        self.__game_state["fullmove"] += 1 if self.__game_state["turn"] == 1 else 0
        if dst_piece == 6 or dst_piece == -6:
            self.__game_state["terminal"] = True
            self.__game_state["winner"] = self.__game_state["turn"]
        # prevent neverending games
        elif self.__game_state["halfmove"] >= 100:
            self.__game_state["terminal"] = True
            self.__game_state["winner"] = 0
        # check if draw by 50 move rule
        elif self.__game_state["fiftymoverule"] >= 50:
            # game is a draw if no pawn was captured or no pawn was moved in the last 50 moves
            self.__game_state["terminal"] = True
            self.__game_state["winner"] = 0
        else:
            # game is not over
            self.__game_state["terminal"] = False
            self.__game_state["winner"] = 0
            # update 50 move rule
            if dst_piece != 0 or src_piece == 1 or src_piece == -1:
                self.__game_state["fiftymoverule"] = 0
            else:
                self.__game_state["fiftymoverule"] += 1

    @property
    def black(self) -> np.uint64:
        return self.__game_state["-1"] | self.__game_state["-2"] | \
            self.__game_state["-3"] | self.__game_state["-4"] | \
            self.__game_state["-5"] | self.__game_state["-6"]

    @property
    def white(self) -> np.uint64:
        return self.__game_state["1"] | self.__game_state["2"] | \
            self.__game_state["3"] | self.__game_state["4"] | \
            self.__game_state["5"] | self.__game_state["6"]

    @property
    def occupied(self) -> np.uint64:
        return self.black | self.white

    @property
    def empty(self) -> np.uint64:
        return ~self.occupied

    @property
    def player_pieces(self) -> np.uint64:
        if self.__game_state["turn"] == 1:
            return self.white
        return self.black

    @property
    def player_king(self) -> np.uint64:
        if self.__game_state["turn"] == 1:
            return self.__game_state["6"].item()
        return self.__game_state["-6"].item()

    @property
    def enemy_or_empty(self) -> np.uint64:
        return ~self.player_pieces

    @property
    def id(self) -> str:
        return self.__game_state.tobytes()

    @property
    def fields(self):
        return self.__game_state.dtype.fields

    @property
    def is_terminal(self) -> bool:
        return self.__game_state["terminal"]

    @property
    def player_turn(self) -> int:
        return self.__game_state["turn"]

    @property
    def win(self) -> int:
        return self.__game_state["winner"]

    @property
    def board(self) -> np.ndarray:
        return self.__board

    @property
    def en_passant_target(self) -> int:
        return self.__game_state["en_passant_target"]
    
    @property
    def castle_rights(self) -> int:
        return self.__game_state["castle_rights"]
    
    @property
    def halfmove(self) -> int:
        return self.__game_state["halfmove"]
    
    @property
    def fullmove(self) -> int:
        return self.__game_state["fullmove"]
    
    @property
    def fifty_move_rule(self) -> int:
        return self.__game_state["fiftymoverule"]
