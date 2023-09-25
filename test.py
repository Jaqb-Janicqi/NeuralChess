from chess_game import Chess
from time import perf_counter

from chess_game import print_bitboard
import chess as chesslib
chess = Chess()
# chess.set_state(chess.fenn_decode(''))
# chess.set_state(chess.fenn_decode('8/8/8/8/8/8/8/r1R2K2 w - - 0 1'))
# chess.set_state(chess.fenn_decode('r6r/1b2k1bq/8/8/7B/8/8/R3K2R b KQ - 3 2')) #d1 n8    
# chess.set_state(chess.fenn_decode('8/8/8/2k5/2pP4/8/B7/4K3 b - d3 0 3'))    #d1 n8
# chess.set_state(chess.fenn_decode('r1bqkbnr/pppppppp/n7/8/8/P7/1PPPPPPP/RNBQKBNR w KQkq - 2 2'))    #d1 n19
# chess.set_state(chess.fenn_decode('r3k2r/p1pp1pb1/bn2Qnp1/2qPN3/1p2P3/2N5/PPPBBPPP/R3K2R b KQkq - 3 2'))   #d1 n5
# chess.set_state(chess.fenn_decode('2kr3r/p1ppqpb1/bn2Qnp1/3PN3/1p2P3/2N5/PPPBBPPP/R3K2R b KQ - 3 2'))   #d1 n44
# chess.set_state(chess.fenn_decode('rnb2k1r/pp1Pbppp/2p5/q7/2B5/8/PPPQNnPP/RNB1K2R w KQ - 3 9')) #d1 n39
# chess.set_state(chess.fenn_decode('2r5/3pk3/8/2P5/8/2K5/8/8 w - - 5 4'))# d1 n9
# chess.set_state(chess.fenn_decode('r1bqkbnr/1ppppppp/8/p7/8/N7/PPPPPPPP/1RBQKBNR b Kkq - 3 3')) #d1 n20
# chess.set_state(chess.fenn_decode(''))
# chess.set_state(chess.fenn_decode(''))
# chess.set_state(chess.fenn_decode(''))


# chess.print_board()
# print_bitboard(chess.state.occupied)
# print(chess.perft(2, chess.state))

tic = perf_counter()
perft1 = chess.perft_debug(1, chess.state)
toc = perf_counter()
print("Perft 1: ", perft1, " in ", toc-tic, " seconds, ", perft1/(toc-tic), " nps")
tic = perf_counter()
perft2 = chess.perft_debug(2, chess.state)
toc = perf_counter()
print("Perft 2: ", perft2, " in ", toc-tic, " seconds, ", perft2/(toc-tic), " nps")
tic = perf_counter()
perft3 = chess.perft_debug(3, chess.state)
toc = perf_counter()
print("Perft 3: ", perft3, " in ", toc-tic, " seconds, ", perft3/(toc-tic), " nps")
tic = perf_counter()
perft4 = chess.perft_debug(4, chess.state)
toc = perf_counter()
print("Perft 4: ", perft4, " in ", toc-tic, " seconds, ", perft4/(toc-tic), " nps")
tic = perf_counter()
perft5 = chess.perft_debug(5, chess.state)
toc = perf_counter()
print("Perft 5: ", perft5, " in ", toc-tic, " seconds, ", perft5/(toc-tic), " nps")
tic = perf_counter()
perft6 = chess.perft_debug(6, chess.state)
toc = perf_counter()
print("Perft 6: ", perft6, " in ", toc-tic, " seconds, ", perft6/(toc-tic), " nps")
tic = perf_counter()
perft7 = chess.perft_debug(7, chess.state)
toc = perf_counter()
print("Perft 7: ", perft7, " in ", toc-tic, " seconds, ", perft7/(toc-tic), " nps")


# legal_moves = chess.get_legal_moves(chess.state)
# # sort legal moves as a string in ascending order
# legal_moves = sorted(legal_moves, key=lambda x: x.uci_str())
# for move in legal_moves:
#     print(move.uci_str())
# print(len(legal_moves))

# (checkmask, pinmask_hv, pinmask_diag, seen_mask) = chess.get_masks(chess.state)
# print_bitboard(checkmask)
# print_bitboard(pinmask_hv)
# print_bitboard(pinmask_diag)
# print_bitboard(seen_mask)

