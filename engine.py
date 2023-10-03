import select
import sys
import chess
import torch

from actionspace import ActionSpace
from cache_read_priority import Cache
from mcts import MCTS
from resnet import ResNet
from state import State


class Engine():
    def __init__(self, args, device="cpu") -> None:
        self.__args = args
        self.__device = device
        self.__action_space = ActionSpace()
        # self.__model = ResNet(
        #     self.__args['num_blocks'], self.__args['num_channels'], self.__action_space.size, device)
        # self.__model.load_state_dict(torch.load(
        #     self.__args['model_path'], map_location=device))
        # self.__model.eval()
        # self.__mcts = MCTS(
        #     self.__args, self.__action_space, Cache(), self.__model)
        self.__mcts = MCTS(self.__args, self.__action_space, Cache())

        self.__default_go_args = {
            "ponder_move": None,
            "wtime": 0,
            "btime": 0,
            "winc": 0,
            "binc": 0,
            "depth": 0,
            "nodes": args["num_searches"],
            "movestogo": 0,
            "movetime": 0,
            "infinite": False
        }
        go_args = self.__default_go_args.copy()

    def get_best_move(self) -> chess.Move:
        return self.__mcts.get_best_move()

    def search(self, fen=None) -> None:
        pass

    def ponder(self, fen_move) -> None:
        move = chess.Move.from_uci(fen_move)
        self.__mcts.restrict_root([move])
        input_buffer = [sys.stdin]

        # continue ponder search until the engine is interrupted
        while True:
            # check input buffer
            ready = select.select(input_buffer, [], [], 0)[0]
            if ready:
                input_str = sys.stdin.readline().strip()
                if input_str == "ponderhit":
                    # user played the ponder move, transition to normal search
                    self.__mcts.select_restricted_as_new_root()
                    break
                elif input_str.startswith("position"):
                    fen = "".join(input_str.split(maxsplit=1)[1:])
                    self.__mcts.root_from_fen(fen)
                    break
            else:
                self.__mcts.search_step()
        self.search()

    def __uci_go(self, go_command) -> None:
        go_args = self.__default_go_args.copy()
        go_command_args = go_command.split()[1:]
        for i in range(0, len(go_command_args), 2):
            if go_command_args[i] == "ponder":
                go_args["ponder_move"] = chess.Move.from_uci(
                    go_command_args[i+1])

            elif go_command_args[i] == "searchmoves":
                moves_str = go_command_args[i+1].split()
                moves = []
                for move_str in moves_str:
                    moves.append(chess.Move.from_uci(move_str))
                self.__mcts.restrict_root(moves)

            elif go_command_args[i] == "wtime":
                go_args["wtime"] = int(go_command_args[i+1])

            elif go_command_args[i] == "btime":
                go_args["btime"] = int(go_command_args[i+1])

            elif go_command_args[i] == "winc":
                go_args["winc"] = int(go_command_args[i+1])

            elif go_command_args[i] == "binc":
                go_args["binc"] = int(go_command_args[i+1])

            elif go_command_args[i] == "depth":
                go_args["depth"] = int(go_command_args[i+1])

            elif go_command_args[i] == "nodes":
                go_args["nodes"] = int(go_command_args[i+1])

            elif go_command_args[i] == "movestogo":
                go_args["movestogo"] = int(go_command_args[i+1])

            elif go_command_args[i] == "movetime":
                go_args["movetime"] = int(go_command_args[i+1])

            elif go_command_args[i] == "infinite":
                go_args["infinite"] = True

    def uci_read(self) -> None:
        print("readok")
        while True:
            try:
                command = input()
                if command == "quit":
                    break

                elif command == "isready":
                    print("readyok")

                elif command == "ucinewgame":
                    self.__mcts.reset()

                elif command.startswith("position"):
                    fen = "".join(command.split(maxsplit=1)[1:])
                    self.__mcts.root_from_fen(fen)

                elif command.startswith("go"):
                    best_move, ponder = self.uci_go(command)
                    print("bestmove", best_move, "ponder", ponder)
            except (KeyboardInterrupt, EOFError):
                break


"position r1qr1b2/1R3pkp/3p2pN/ppnPp1Q1/bn2P3/4P2P/pBBP2P1/5RK1 w - - 0 1"
args = {"num_searches": 2000, "c_puct": 1}
engine = Engine(args)
engine.uci_read()
