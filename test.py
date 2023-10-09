import chess
from state import State
import cProfile
import microdict
from microdict import run_tests

# fen = "rnbqkbnr/pppp1ppp/8/p7/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
# print(fen)

# def get_state(fen):
#     for i in range(4000):
#         board = chess.Board(fen)
#         state = State(board)

# with cProfile.Profile() as pr:
#     get_state(fen)
# pr.print_stats()


# run_tests.run()

board = chess.Board()
state = State(board)
print(state.outcome)