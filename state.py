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
    def id(self):
        """Returns a unique id str for the current state."""

        kings = self.__board.kings
        queens = self.__board.queens
        rooks = self.__board.rooks
        bishops = self.__board.bishops
        knights = self.__board.knights
        pawns = self.__board.pawns
        fullmove = self.__board.fullmove_number
        halfmove = self.__board.halfmove_clock
        ep_square = self.__board.ep_square
        white_mask = self.__board.occupied_co[chess.WHITE]
        black_mask = self.__board.occupied_co[chess.BLACK]
        turn = self.__board.turn
        castling = self.__board.castling_rights
        s_id = np.array([kings, queens, rooks, bishops, knights, pawns, fullmove,
                        halfmove, ep_square, white_mask, black_mask, turn, castling])
        return s_id.tobytes()

    @property
    def turn(self) -> bool:
        return self.__board.turn

    @property
    def encoded(self) -> np.ndarray:
        # transform position to white perspecitve if necessary
        board = self.__board if self.__board.turn else self.__board.mirror()
        # encode board
        encoded = np.zeros((8, 8, 11), dtype=np.float16)
        for i in range(64):
            piece = board.piece_at(i)
            if piece is not None:
                encoded[i // 8, i % 8, piece.piece_type - 1 +
                        6 * piece.color] = 1
        # encode castling rights
