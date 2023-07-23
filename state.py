import numpy as np


class State():
    def __init__(self, bitboards, board, turn, winner=None) -> None:
        self.__bitboards: np.ndarray = bitboards
        self.__board: np.ndarray = board
        self.__turn: np.byte = turn
        self.__winner: np.byte = winner

    def encode(self):
        if self.__turn == 1:
            return self.__board
        return np.flip(-self.__board, axis=(0, 1))

    @property
    def id(self) -> int:
        return self.__bitboards.tobytes() + self.__turn.tobytes()

    @property
    def is_terminal(self) -> bool:
        return self.__winner is not None

    @property
    def win(self) -> np.byte:
        return self.__winner
    
    @property
    def board(self) -> np.ndarray:
        return self.__board
    
    @property
    def turn(self) -> np.byte:
        return self.__turn
