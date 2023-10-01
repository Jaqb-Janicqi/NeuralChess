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

        # uci variables
        self.__wtime = 0
        self.__btime = 0
        self.__winc = 0
        self.__binc = 0
        self.__depth = 0
        self.__nodes = args["num_searches"]
        self.__movestogo = 0
        self.__movetime = 0
        self.__infinite = False

    def get_move(self, fen) -> chess.Move:
        self.__mcts.reset()
        self.__mcts.search(fen)
        return self.__mcts.get_best_move()
    
    def get_best_move(self) -> chess.Move:
        return self.__mcts.get_best_move()

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
                    go_args = command.split()[1:]
                    for i in range(0, len(go_args), 2):
                        if go_args[i] == "wtime":
                            self.__wtime = int(go_args[i+1])
                        elif go_args[i] == "btime":
                            self.__btime = int(go_args[i+1])
                        elif go_args[i] == "winc":
                            self.__winc = int(go_args[i+1])
                        elif go_args[i] == "binc":
                            self.__binc = int(go_args[i+1])
                        elif go_args[i] == "depth":
                            self.__depth = int(go_args[i+1])
                        elif go_args[i] == "nodes":
                            self.__nodes = int(go_args[i+1])
                        elif go_args[i] == "movestogo":
                            self.__movestogo = int(go_args[i+1])
                        elif go_args[i] == "movetime":
                            self.__movetime = int(go_args[i+1])
                        elif go_args[i] == "infinite":
                            self.__infinite = True
                    if self.__infinite:
                                while True:
                                    rlist, _, _ = select.select([sys.stdin], [], [], 0)
                                    if rlist:
                                        pass
                        
                    # print("bestmove", move)
            except (KeyboardInterrupt, EOFError):
                break

"position r1qr1b2/1R3pkp/3p2pN/ppnPp1Q1/bn2P3/4P2P/pBBP2P1/5RK1 w - - 0 1"
args = {"num_searches": 2000, "c_puct": 1}
engine = Engine(args)
engine.uci_read()
