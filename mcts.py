import time
from math import sqrt

import chess
import numpy as np

from actionspace import ActionSpace
from cache_read_priority import Cache
from resnet import ResNet
from state import State


class Node():
    def __init__(self, args, state, action_taken=None, parent=None, policy_value=0) -> None:
        self.args = args
        self.parent: "Node" = parent
        self.children: list = []
        self.state: State = state
        self.action_taken: chess.Move = action_taken
        self.value: float = 0
        self.policy_value: float = policy_value
        self.visit_count: int = 0

    def expand(self, policy, actions, nodes_dict) -> None:
        """
        Expand the node with the given policy and actions. 
        Add new nodes to nodes_dict.
        """

        for prob, action in zip(policy, actions):
            next_state = self.state.next_state(action)

            # to narrow the search space, connect branches that lead to the same state
            fen = next_state.fen
            if fen in nodes_dict:
                self.children.append(nodes_dict[fen])
            else:
                new_node = Node(
                    self.args, next_state, action_taken=action, parent=self, policy_value=prob)
                nodes_dict[fen] = new_node
                self.children.append(new_node)

    def set_parent(self, parent) -> None:
        """Set the parent of the node"""

        self.parent = parent

    def select(self) -> "Node":
        """Select the child node with the highest UCB value, """

        # select child with best ucb value
        best_child = max(self.children, key=lambda child: child.ucb)
        # since branches of the tree are connected,
        # update the parent to allow for backpropagation
        best_child.set_parent(self)
        return best_child

    def backpropagate(self, value) -> None:
        """Backpropagate the value of the node to the root"""

        self.visit_count += 1
        self.value += value
        if self.parent:
            self.parent.backpropagate(-value)

    def simulate(self) -> int:
        """Simulate a game from the current state"""

        # since a player cannot checkmate himself,
        # only the player who has just moved can be the winner
        # therefore, distinction who won is not necessary
        if self.state.is_checkmate:
            return 1
        return 0

    @property
    def ucb(self) -> float:
        """Calculate the UCB value of the node"""

        q = self.value / (1 + self.visit_count)
        if self.parent is None:
            u = 0
        else:
            u = self.policy_value * self.args["c_puct"] * \
                sqrt(self.parent.visit_count) / (1 + self.visit_count)
        return q + u


class MCTS():
    """Monte Carlo Tree Search with optional ResNet model and evaluation cache"""

    def __init__(self, args, action_space, cache, model=None) -> None:
        self.args = args
        self.__model: ResNet = model
        self.__root: Node = None
        self.__nodes: dict = dict()
        self.__action_space: ActionSpace = action_space
        self.__cache: Cache = cache
        self.__uniform_policy = np.ones(
            self.__action_space.size, dtype=np.float32) / self.__action_space.size
        self.__depth_reached = 0

    def calculate_policy(self, node: Node, legal_moves: list) -> np.ndarray:
        """Calculate the policy and value of a given node"""

        policy, policy_value = self.__model(
            self.__model.get_tensor_state(node.state.encoded))
        policy = self.__model.get_policy(policy)
        policy_value = self.__model.get_value(policy_value)

        # calculate legal policy
        legal_ids = np.array(
            [self.__action_space.get_key(move) for move in legal_moves])
        legal_policy = np.zeros(self.__action_space.size)
        legal_policy[legal_ids] = policy[legal_ids]
        legal_policy = legal_policy[legal_policy != 0]
        legal_policy = legal_policy / np.sum(legal_policy)
        return legal_policy, policy_value

    def search_step(self, max_depth: int) -> None:
        """Perform a single tree search step"""

        # initialize node to root
        node = self.__root
        # select a leaf node

        depth_reached = 0
        for depth in range(max_depth):
            if not node.children:
                depth_reached = depth
                break
            node = node.select()
        value = node.simulate()
        legal_moves = node.state.legal_moves

        # expand the node if it is not terminal
        if not value:
            # use uniform policy if no model is provided
            if self.__model:
                fen = node.state.fen
                # check cache for policy, skip calculation if found
                legal_policy = self.__cache[fen]
                if legal_policy is None:
                    legal_policy, value = self.calculate_policy(
                        node, legal_moves)
                    # add policy cache
                    self.__cache[fen] = legal_policy
            else:
                # get uniform policy
                legal_policy = self.__uniform_policy
            node.expand(legal_policy, legal_moves, self.__nodes)
        node.backpropagate(value)

        # update depth reached
        if depth_reached > self.__depth_reached:
            self.__depth_reached = depth_reached
        return depth_reached

    def get_dist(self) -> np.ndarray:
        """Return the distribution of visits of leaf nodes"""

        dist = np.zeros(self.__action_space.size)
        for child in self.__root.children:
            dist[self.__action_space.get_key(
                child.action_taken)] = child.visit_count
        return dist / np.sum(dist)

    def initialize_root(self, reinitialize=False) -> None:
        """Expand root node if necessary"""

        if reinitialize:
            self.__root.children.clear()
        if self.__root.state.is_terminal or self.__root.children:
            return

        if self.__model:
            legal_moves = self.__root.state.legal_moves
            legal_policy, value = self.calculate_policy(
                self.__root, legal_moves)
            self.__root.expand(legal_policy, legal_moves, self.__nodes)
        else:
            self.__root.expand(self.__uniform_policy,
                               self.__root.state.legal_moves, self.__nodes)

    def set_root(self, state: State) -> None:
        """Set the root node to the given state"""

        if state.fen in self.__nodes:
            self.__root = self.__nodes[state.fen]
        else:
            self.__root = Node(self.args, state)
            self.__nodes[state.fen] = self.__root
        # reset depth reached
        self.__depth_reached = 0

    def root_from_fen(self, fen: str) -> None:
        """Set the root node to the given fen"""

        board = chess.Board(fen)
        state = State(board)
        self.set_root(state)
        self.initialize_root()

    def restrict_root(self, actions: list[chess.Move]) -> None:
        """Restrict the root node to the given actions"""

        # get the child node corresponding to the action
        for child in self.__root.children:
            # remove children not specified in actions
            if child.action_taken not in actions:
                self.root.children.remove(child)
                break

    def select_restricted_as_new_root(self) -> None:
        """Select the child node as new root, provided that the root has one child"""

        if len(self.__root.children) == 1:
            self.__root = self.__root.children[0]
            self.__root.set_parent(None)

    def search(self, max_depth: int, num_searches=None) -> tuple:
        """
        Perform a search for a given number of searches up to given depth. 
        num_searches=args['num_searches'] by default
        """

        if not num_searches:
            num_searches = self.args["num_searches"]

        # perform search
        for _ in range(num_searches):
            self.search_step(max_depth)
        return (self.__root.state.encoded, self.get_dist(), -self.__root.value/self.__root.visit_count)

    def best_move(self) -> tuple:
        """Return the best move and the ponder move"""

        bestmove = max(self.__root.children,
                       key=lambda child: child.visit_count)
        pondermove = max(bestmove.children,
                         key=lambda child: child.visit_count)
        return (bestmove.action_taken.uci(), pondermove.action_taken.uci())

    def reset(self) -> None:
        """Reset the tree and associated caches"""

        self.__root = None
        self.__cache.clear()
        self.__nodes.clear()

    def reset_node_cache(self) -> None:
        """Reset the node cache"""

        self.__nodes.clear()

    @property
    def node_count(self) -> int:
        """Return the number of nodes in the tree"""

        return len(self.__nodes)

    @property
    def root_fen(self) -> str:
        """Return the fen of the root node"""

        return self.__root.state.fen

    @property
    def evaluation(self) -> float:
        """Return the evaluation of the root node"""

        return -self.__root.value/self.__root.visit_count

    @property
    def depth(self) -> int:
        """Return the maximum depth reached in the tree"""

        return self.__depth_reached
