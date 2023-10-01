import chess
import numpy as np


class State():
    def __init__(self, board) -> None:
        self.__board: chess.Board = board

    def push(self, action) -> None:
        self.__board.push(action)

    def copy(self) -> "State":
        return State(self.__board.copy(stack=False))

    def next_state(self, action) -> chess.Board:
        new_state = self.copy()
        new_state.push(action)
        return new_state

    @property
    def is_checkmate(self) -> bool:
        return self.__board.is_checkmate()
    
    @property
    def is_terminal(self) -> bool:
        return self.__board.is_game_over()

    @property
    def legal_moves(self) -> list:
        return self.__board.legal_moves
    
    @property
    def fen(self) -> str:
        return self.__board.fen()

    @property
    def encoded(self) -> np.ndarray:
        # transform position to white perspecitve if necessary
        board = self.__board if self.__board.turn else self.__board.mirror()
        # encode board
