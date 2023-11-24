import numpy as np
import chess
from actionspace import ActionSpace as asp


def bb_to_matrix(bb: np.uint64) -> np.ndarray:
    """Converts a bitboard to a 8x8 matrix."""

    return np.unpackbits(np.frombuffer(bb.tobytes(), dtype=np.uint8)).astype(np.float32).reshape(8, 8)


class Node():
    def __init__(self, c_puct, state, action_space, action=None, parent=None) -> None:
        self.__c_puct: np.float32 = np.float32(c_puct)
        self.__state: chess.Board = state
        self.__action_space: asp = action_space
        self.__action: int = action
        self.__parent: Node = parent
        self.__children = {}  # {action: Node}
        self.__is_expanded: bool = False
        self.__id = None

        self.__child_visit_count = np.zeros(1968, dtype=np.float32)
        self.__child_value = np.zeros(1968, dtype=np.float32)
        self.__child_virtual_loss = np.zeros(1968, dtype=np.float32)
        self.__policy = np.zeros(1968, dtype=np.float32)
        self.__legal_mask = np.full(1968, -np.inf, dtype=np.float32)

    def add_child(self, move: chess.Move) -> None:
        child_state = self.__state.copy()
        child_state.push(move)
        action = self.__action_space.get_key(move)
        self.__children[action] = Node(
            self.__c_puct, child_state, self.__action_space, action, self)

    def expand(self, policy) -> None:
        self.__is_expanded = True
        self.__policy = policy
        for move in self.legal_moves:
            key = self.__action_space.get_key(move)
            self.__legal_mask[key] = 0

    def apply_virtual_loss(self) -> None:
        if self.__parent:
            self.__parent.child_virtual_loss[self.__action] = -np.inf

    def revert_virtual_loss(self) -> None:
        if self.__parent:
            self.__parent.child_virtual_loss[self.__action] = 0

    def select(self) -> 'Node':
        best_child = self.best_child
        if not best_child in self.__children:
            self.add_child(self.__action_space[best_child])
        return self.__children[best_child]

    def backpropagate(self, value) -> None:
        if self.__parent:
            self.value += value
            self.__parent.backpropagate(value)

    @property
    def child_visit_count(self):
        return self.__child_visit_count

    @property
    def child_value(self):
        return self.__child_value

    @property
    def child_virtual_loss(self):
        return self.__child_virtual_loss

    @property
    def visit_count(self):
        if self.__parent is None:
            return 1
        return self.__parent.child_visit_count[self.__action]

    @visit_count.setter
    def visit_count(self, value):
        self.__parent.child_visit_count[self.__action] = value

    @property
    def value(self):
        return self.__parent.child_value[self.__action]

    @value.setter
    def value(self, value):
        self.__parent.child_value[self.__action] = value

    @property
    def children_q(self):
        return self.__child_value / (1 + self.__child_visit_count)

    @property
    def children_u(self):
        return np.sqrt(self.visit_count) * self.__c_puct * (
            self.__policy / (1 + self.__child_visit_count))

    @property
    def best_child(self):
        return np.argmax(self.children_q + self.children_u + self.__legal_mask)

    @property
    def is_checkmate(self) -> bool:
        return self.__state.is_checkmate()

    @property
    def is_terminal(self) -> bool:
        return self.__state.is_game_over(True)

    @property
    def legal_moves(self) -> list:
        return self.__state.legal_moves

    @property
    def fen(self) -> str:
        return self.__state.fen()

    @property
    def id(self):
        """Returns a unique id str for the current state."""

        if self.__id:
            return self.__id
        kings = self.__state.kings
        queens = self.__state.queens
        rooks = self.__state.rooks
        bishops = self.__state.bishops
        knights = self.__state.knights
        pawns = self.__state.pawns
        fullmove = self.__state.fullmove_number
        halfmove = self.__state.halfmove_clock
        ep_square = self.__state.ep_square
        white_mask = self.__state.occupied_co[chess.WHITE]
        black_mask = self.__state.occupied_co[chess.BLACK]
        turn = self.__state.turn
        castling = self.__state.castling_rights
        s_id = np.array([kings, queens, rooks, bishops, knights, pawns, fullmove,
                        halfmove, ep_square, white_mask, black_mask, turn, castling])
        self.__id = s_id.tobytes()
        return self.__id

    @property
    def turn(self) -> bool:
        return self.__state.turn

    @property
    def encoded(self) -> np.ndarray:
        """Returns a 9x8x8 stack of matrices, representing the current state from white perspective."""

        white = self.__state.occupied_co[chess.WHITE]
        black = self.__state.occupied_co[chess.BLACK]
        p_color = 1 if self.__state.turn else -1
        # get board attributes
        castling_bb = np.uint64(self.__state.castling_rights)
        ep_square_bb = self.__state.ep_square
        if not ep_square_bb:
            ep_square_bb = 0
        # get pieces
        piece_bbs = [
            self.__state.kings,
            self.__state.queens,
            self.__state.rooks,
            self.__state.bishops,
            self.__state.knights,
            self.__state.pawns
        ]
        matrices = []
        # create color matrix
        color_matrix = np.full((8, 8), p_color, dtype=np.float32)
        matrices.append(color_matrix)
        # convert pieces
        for piece in range(6):
            white_pieces = piece_bbs[piece] & white
            white_pieces = bb_to_matrix(np.uint64(white_pieces))
            black_pieces = piece_bbs[piece] & black
            black_pieces = bb_to_matrix(np.uint64(black_pieces))
            black_pieces *= -1
            matrices.append(white_pieces + black_pieces)
        # create ep_square matrix
        matrices.append(bb_to_matrix(np.uint64(ep_square_bb)))
        # convert castling
        matrices.append(bb_to_matrix(np.uint64(castling_bb)))
        return np.array(matrices, np.float32)

    @property
    def is_expanded(self) -> bool:
        return self.__is_expanded

    @child_value.setter
    def child_value(self, action, value):
        self.child_value[action] = value

    @property
    def children(self):
        return self.__children

    @property
    def action(self):
        return self.__action
    
    @property
    def turn(self):
        return self.__state.turn
