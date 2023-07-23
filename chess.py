import numpy as np

from actionspace import ActionSpace
from move import Move
from state import State


class Chess():
    def __init__(self, args, board_size, actionspace) -> None:
        self.__args = args
        self.__board_size: int = board_size
        self.__state: State = self.load_default(board_size)
        self.__action_space: ActionSpace = actionspace

    def square_index(self, rank, file) -> np.ubyte:
        return np.ubyte(rank * self.__board_size + file)

    # def bind_move(self, rank, file, board_size) -> tuple[int, int]:
    #     rank = max(0, min(rank, board_size - 1))
    #     file = max(0, min(file, board_size - 1))
    #     return rank, file

    # def is_in_bounds(self, rank, file, board_size) -> bool:
    #     return 0 <= file < board_size and 0 <= rank < board_size

    def get_moves(self, state: State) -> list[Move]:
        pass

    def next_game(self) -> None:
        self.__state = self.load_default(self.__board_size)

    def next_state(self, state: State, move: Move) -> State:
        src_y, src_x = move["src_y"], move["src_x"]
        dst_y, dst_x = move["dst_y"], move["dst_x"]
        src_square = state.board[src_y, src_x]
        dst_square = state.board[dst_y, dst_x]
        new_board = state.board.copy()
        winner = None
        if np.absolute(src_square) == np.ubyte(0):
            raise Exception("Invalid action: source piece is empty")
        if np.absolute(dst_square) == np.ubyte(6):
            winner = int(src_piece * state.turn)
        if move["promo_piece"] != np.ubyte(0):
            src_piece = move["promo_piece"] * state.turn
        elif move["special_dir"] != np.ubyte(0):
            rank = 0 if state.turn == 1 else self.__board_size - 1
            if move["castle_dir"] == np.ubyte(1):
                new_board[rank, 5] = new_board[rank, 7]
                new_board[rank, 7] = 0
            else:
                new_board[rank, 3] = new_board[rank, 0]
                new_board[rank, 0] = 0
        new_board[src_y, src_x] = 0
        new_board[dst_y, dst_x] = src_piece
        if np.count_nonzero(new_board) == 2:
            winner = 0
        return State(new_board, -state.turn, winner)

    def fenn_decode(self, fenn_string, board_size: int) -> State:
        bitboard_offset = {
            'P': 0,
            'N': 1,
            'B': 2,
            'R': 3,
            'Q': 4,
            'K': 5,
            'p': 6,
            'n': 7,
            'b': 8,
            'r': 9,
            'q': 10,
            'k': 11,
            'moved_or_not': 12,
            'en_passant': 13,
        }
        piece_id = {
            'P': 1,
            'N': 2,
            'B': 3,
            'R': 4,
            'Q': 5,
            'K': 6,
            'p': -1,
            'n': -2,
            'b': -3,
            'r': -4,
            'q': -5,
            'k': -6,
        }
        bitboards = np.zeros(len(bitboard_offset), dtype=np.uint64)
        board = np.zeros((board_size, board_size), dtype=np.byte)
        player_turn = np.byte(1)
        state = State(bitboards, board, player_turn)
        file = 0
        rank = board_size - 1
        for char in fenn_string:
            if char == '/':
                pass
            elif char.isdigit():
                file += int(char)
                r, file = np.divmod(file, board_size)
                rank -= r
            elif char in bitboard_offset:
                bitboard_id = bitboard_offset[char]
                board[rank, file] = piece_id[char]
                bitboards[bitboard_id] |= np.uint64(
                    1) << self.square_index(rank, file)
                bitboards[bitboard_offset['moved_or_not']] |= np.uint64(
                    1) << self.square_index(rank, file)
                file += 1
                r, file = np.divmod(file, board_size)
                rank -= r
            else:
                raise ValueError(f'Unsupported character {char} in FEN string')
        return state

    def load_default(self, board_size: int) -> State:
        return self.fenn_decode('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR', board_size)

    @property
    def action_space(self):
        return self.__action_space
