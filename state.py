import numpy as np


class State():
    def __init__(self, game_state) -> None:
        self.__game_state: np.ndarray = game_state

    def encode(self):
        return self.__game_state["board"] if self.__game_state["turn"] == 1 else np.flip(-self.__game_state["board"], axis=(0, 1))

    @property
    def id(self) -> str:
        return self.__game_state.tobytes()

    @property
    def is_terminal(self) -> bool:
        return self.__game_state["terminal"] == True

    @property
    def win(self) -> np.byte:
        return self.__game_state["winner"]

    @property
    def board(self) -> np.ndarray:
        return self.__game_state["board"]

    @property
    def turn(self) -> np.byte:
        return self.__game_state["turn"]
