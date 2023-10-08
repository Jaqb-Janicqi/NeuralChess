from math import sqrt

import chess
import numpy as np

from actionspace import ActionSpace
from cache_read_priority import Cache
from resnet import ResNet
from state import State


class Node():
    def __init__(self, args, parent=None, state=None, action_taken=None, policy_value=0) -> None:
        self.__args = args
        self.__parent: Node = parent
        self.__children: list = []
        self.__state: State = state
        self.__action_taken: chess.Move = action_taken
        self.__value: float = 0
        self.__policy_value: float = policy_value
        self.__visit_count: int = 0

    def load_state(self) -> None:
        if not self.__state:
            self.__state = self.__parent.state.next_state(self.__action_taken)

    def expand(self, policy, actions, nodes_dict) -> None:
        """
        Expand the node with the given policy and actions. 
        Add new nodes to nodes_dict.
        """

        for prob, action in zip(policy, actions):
            new_node = Node(args=self.__args, parent=self,
                            action_taken=action, policy_value=prob)

            # to narrow the search space, connect branches that lead to the same state
            node_id = new_node.state.id
            if node_id in nodes_dict:
                self.__children.append(nodes_dict[node_id])
            else:
                nodes_dict[node_id] = new_node
                self.__children.append(new_node)

    def select(self) -> "Node":
        """Select the child node with the highest UCB value, """

        # select child with best ucb value
        best_child = max(
            self.__children, key=lambda child: child.ucb(self))
        return best_child

    def backpropagate(self, value: float) -> None:
        """Backpropagate the value of the node to the root"""

        self.__visit_count += 1
        self.__value += value
        if self.__parent:
            self.__parent.backpropagate(-value)

    def simulate(self) -> int:
        """Simulate a game from the current state"""

        # since a player cannot checkmate himself,
        # only the player who has just moved can be the winner
        # therefore, distinction who won is not necessary
        if self.__state.is_checkmate:
            return 1
        if self.__state.is_terminal:
            return 0
        return None

    def clear_children(self) -> None:
        """Clear the children of the node"""

        self.__children.clear()

    def remove_child(self, child: "Node") -> None:
        """Remove a child from the node"""

        self.__children.remove(child)

    def ucb(self, parent: "Node") -> float:
        """Calculate the UCB value of the node"""

        q = self.__value / (1 + self.__visit_count)
        if parent is None:
            u = 0
        else:
            u = self.__policy_value * self.__args["c_puct"] * \
                sqrt(parent.visit_count) / (1 + self.__visit_count)
        return q + u

    @property
    def action_taken(self) -> chess.Move:
        """Return the action taken to reach the node"""

        return self.__action_taken

    @property
    def visit_count(self) -> int:
        """Return the number of visits of the node"""

        return self.__visit_count

    @property
    def value(self) -> float:
        """Return the value of the node"""

        return self.__value

    @property
    def state(self) -> State:
        """Return the state of the node"""

        self.load_state()
        return self.__state

    @property
    def children(self) -> list:
        """Return the children of the node"""

        return self.__children


class MCTS():
    """Monte Carlo Tree Search with optional ResNet model and evaluation cache"""

    def __init__(self, args, action_space, cache, model=None) -> None:
        self.__args = args
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
        if self.__depth_reached < depth_reached:
            self.__depth_reached = depth_reached
        value = node.simulate()
        legal_moves = node.state.legal_moves

        # expand the node if it is not terminal
        if value is None:
            # use uniform policy if no model is provided
            if self.__model:
                node_id = node.state.id
                # check cache for policy, skip calculation if found
                legal_policy = self.__cache[node_id]
                if legal_policy is None:
                    legal_policy, value = self.calculate_policy(
                        node, legal_moves)
                    # add policy cache
                    self.__cache[node_id] = legal_policy
            else:
                # get uniform policy
                legal_policy = self.__uniform_policy
            node.expand(legal_policy, legal_moves, self.__nodes)
        node.backpropagate(value)

        # update depth reached
        if depth_reached > self.__depth_reached:
            self.__depth_reached = depth_reached

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
            self.__root.clear_children()

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

        if state.id in self.__nodes:
            self.__root = self.__nodes[state.id]
        else:
            self.__root = Node(args=self.__args, parent=None, state=state)
            self.__nodes[state.id] = self.__root
        # reset depth reached
        self.__depth_reached = 0

    def root_from_fen(self, fen: str) -> None:
        """Set the root node to the given fen"""

        board = chess.Board(fen)
        state = State(board)
        self.set_root(state)
        self.initialize_root()

    def restrict_root(self, actions: list[chess.Move]) -> None:
        """Restrict the root nodes to the given actions"""

        # get the child node corresponding to the action
        for child in self.__root.children:
            # remove children not specified in actions
            if child.action_taken not in actions:
                self.__root.remove_child(child)

    def select_child_as_new_root(self, action: chess.Move) -> None:
        """Select the child node as new root"""

        for child in self.__root.children:
            if child.action_taken == action:
                self.set_root(child.state)
                break

    def best_move(self) -> tuple:
        """Return the best move and the ponder move"""

        if not self.__root.children:
            return None, None
        bestchild = max(self.__root.children,
                        key=lambda child: child.visit_count)
        if not bestchild.children:
            return bestchild.action_taken.uci(), None
        ponderchild = max(bestchild.children,
                          key=lambda child: child.visit_count)
        return (bestchild.action_taken.uci(), ponderchild.action_taken.uci())

    def best_line(self) -> list:
        """Return the best line of the tree"""

        line = []
        node = self.__root
        while node.children:
            node = max(node.children, key=lambda child: child.visit_count)
            line.append(node.action_taken.uci())
        return line

    def reset_node_cache(self) -> None:
        """Reset the node cache"""

        self.__nodes.clear()

    def reset_evaluation_cache(self) -> None:
        """Reset the evaluation cache"""

        self.__cache.clear()

    def reset(self) -> None:
        """Reset the tree and associated caches"""

        self.__root = None
        self.reset_node_cache()
        self.reset_evaluation_cache()

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
        """Return the evaluation of the root node as a value between -1 and 1"""

        return -self.__root.value/self.__root.visit_count

    @property
    def depth(self) -> int:
        """Return the maximum depth reached in the tree"""

        return self.__depth_reached

    @property
    def root(self) -> bool:
        """Return the root node"""

        return self.__root
