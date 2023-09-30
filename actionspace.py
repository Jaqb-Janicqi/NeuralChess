import chess


class ActionSpace():
    def __init__(self) -> None:
        self.__actionspace: dict = dict()
        self.__key_map: dict = dict()
        self.__size: int = 0

    def __getitem__(self, key) -> tuple:
        return self.__actionspace[key]

    def add(self, obj) -> None:
        if obj in self.__key_map:
            return
        self.__actionspace[self.__size] = obj
        self.__key_map[obj] = self.__size
        self.__size += 1

    def get_key(self, obj) -> int:
        return self.__key_map[obj]

    def calculate(self) -> None:
        # Create an empty chess board
        board = chess.Board()
        # Initialize an empty list to store all possible moves
        action_space = []
        # Iterate over all squares and all piece types
        for square in chess.SQUARES:
            for piece_type in chess.PIECE_TYPES:
                for color in chess.COLORS:
                    # Set player to color
                    board.turn = color
                    # Place piece on given square
                    board.set_piece_at(square, chess.Piece(piece_type, color))

                    # If the piece is a pawn, place other pawns around it to include attacks
                    if piece_type == chess.PAWN:
                        other_color = chess.WHITE if color == chess.BLACK else chess.BLACK
                        for square_offset in [-7, -9, 7, 9]:
                            if square + square_offset in chess.SQUARES:
                                board.set_piece_at(
                                    square + square_offset, chess.Piece(piece_type, other_color))

                    for move in board.legal_moves:
                        # If the move is not already in the action space, add it
                        if move not in action_space:
                            action_space.append(move)

                    # Clear the board
                    board.clear()
        action_space = sorted(action_space, key=lambda x: x.uci())

        for action in action_space:
            self.add(action)

    @property
    def size(self) -> int:
        return self.__size
