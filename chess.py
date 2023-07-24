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
    
    def set_bit(value, bit) -> None:
        value |= (1<<bit)

    def clear_bit(value, bit) -> None:
        value &= ~(1<<bit)

    def get_bitboard_id(self, piece_id, turn) -> np.ubyte:
        if turn > 0:
            return piece_id
        return 2 * piece_id

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
        src_sq_id = self.square_index(src_y, src_x)
        dst_sq_id = self.square_index(dst_y, dst_x)
        src_sq = state.board[src_y, src_x]
        dst_sq = state.board[dst_y, dst_x]
        src_id = np.absolute(src_sq)
        dst_id = np.absolute(dst_sq)
        promo_piece = move["promo_piece"]
        special_dir = move["special_dir"]
        turn = state.turn
        winner = None
        new_board = state.board.copy()
        bitboards = state.bitboards.copy()

        if src_id == np.ubyte(0):
            raise Exception("Invalid action: source piece is empty")
        if dst_id == np.ubyte(6):
            
            winner = state.turn
        if move["promo_piece"] != np.ubyte(0):
            
            src_piece = move["promo_piece"] * state.turn

        elif move["special_dir"] != np.ubyte(0):
            if src_id == np.ubyte(1):
                if dst_id == np.ubyte(0):
                    if dst_y == 0 or dst_y == self.__board_size - 1:
                        src_piece = np.ubyte(5) * state.turn
                    else:
                        src_piece = np.ubyte(1) * state.turn
                else:
                    src_piece = np.ubyte(1) * state.turn
            rank = 0 if state.turn == 1 else self.__board_size - 1
            
                
        
        new_board[src_y, src_x] = 0
        new_board[dst_y, dst_x] = src_piece
        if np.count_nonzero(new_board) == 2:
            winner = 0
        return State(new_board, -state.turn, winner)

    def fenn_decode(self, fenn_string, board_size: int) -> State:
        bitboard_offset = {
            'any': 0,
            'P': 1,
            'p': 2,
            'N': 3,
            'n': 4,
            'B': 5,
            'b': 6,
            'R': 7,
            'r': 8,
            'Q': 9,
            'q': 10,
            'K': 11,
            'k': 12,
            'moved_or_not': 13,
            'en_passant': 14,
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
        # display all of the bitboards
        for i in range(len(bitboard_offset)):
            bitboard = np.binary_repr(bitboards[i], width=64)
            index = 0
            print(f'{list(bitboard_offset.keys())[i]}')
            for r in range(board_size):
                for f in range(board_size):
                    print(bitboard[index], end=' ')
                    index += 1
                print()
            print()
        return state

    def load_default(self, board_size: int) -> State:
        return self.fenn_decode('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR', board_size)

    @property
    def action_space(self):
        return self.__action_space


if __name__ == "__main__":
    gm = Chess(dict(), 8, ActionSpace(8, 8))
    st = gm.load_default()