from copy import deepcopy

import numpy as np

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
    ("fullmove", np.uint8)
])


class State():
    def __init__(self) -> None:
        self.__game_state: np.ndarray = np.zeros((), dtype=npstate)
        self.board = np.zeros((8, 8), dtype=np.int8)

    def __getitem__(self, key) -> np.ndarray:
        return self.__game_state[key].item()

    def __setitem__(self, key, value) -> None:
        self.__game_state[key] = value

    def __str__(self) -> str:
        return str(self.__game_state)

    def copy(self):
        return deepcopy(self.__game_state)

    def encode(self):
        board = self.board if self.__game_state["turn"] == 1 else np.flip(-self.board, axis=(0, 1))
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
            return self.__game_state["6"]
        return self.__game_state["-6"]

    @property
    def enemy_or_empty(self) -> np.uint64:
        return ~self.player_pieces

    @property
    def id(self) -> str:
        return self.__game_state.tobytes()

    @property
    def fields(self):
        return self.__game_state.dtype.fields
