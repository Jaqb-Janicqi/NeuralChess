import chess
import numpy as np


def bb_to_matrix(bb: np.uint64) -> np.ndarray:
    """Converts a bitboard to a 8x8 matrix."""

    return np.unpackbits(np.frombuffer(bb.tobytes(), dtype=np.uint8)).astype(np.float16).reshape(8, 8)


class State():
    def __init__(self, board) -> None:
        self.__board: chess.Board = board
        self.__id = None

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

        if self.__id:
            return self.__id

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
        self.__id = s_id.tobytes()
        return self.__id

    @property
    def turn(self) -> bool:
        return self.__board.turn

    @property
    def encoded(self) -> np.ndarray:
        """Returns a 13x8x8 matrix representing the current state, from white perspective."""

        # get board from white perspective
        board = self.__board if self.__board.turn else self.__board.mirror()
        # get board attributes
        castling_bb = np.uint64(board.castling_rights)
        fullmove_bb = self.__board.fullmove_number
        halfmove_bb = self.__board.halfmove_clock
        ep_square_bb = self.__board.ep_square
        if not ep_square_bb:
            ep_square_bb = 0
        # get pieces
        piece_bbs = [
            self.__board.kings,
            self.__board.queens,
            self.__board.rooks,
            self.__board.bishops,
            self.__board.knights,
            self.__board.pawns
        ]
        # convert bitboards to matrices
        matrices = []
        # convert pieces
        for color in range(2):
            for piece in range(6):
                matrices.append(bb_to_matrix(
                    np.uint64(piece_bbs[piece] & board.occupied_co[color])))
        # create ep_square matrix
        matrices.append(bb_to_matrix(np.uint64(ep_square_bb)))
        # convert castling
        matrices.append(bb_to_matrix(np.uint64(castling_bb)))
        # create fullmove matrix
        fullmove_matrix = np.full((8, 8), fullmove_bb, dtype=np.float16)
        matrices.append(fullmove_matrix)
        # create halfmove matrix
        halfmove_matrix = np.full((8, 8), halfmove_bb, dtype=np.float16)
        matrices.append(halfmove_matrix)
        return np.array(matrices)
