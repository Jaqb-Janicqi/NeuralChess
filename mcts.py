import numpy as np
import chess
import torch
from actionspace import ActionSpace as asp
from mcts_node import Node
from cache_read_priority import Cache
from resnet import ResNet


class MCTS():
    def __init__(self, c_puct, action_space, cache, model=None) -> None:
        self.__c_puct: np.float32 = np.float32(c_puct)
        self.__action_space: asp = action_space
        self.__cache: Cache = cache
        self.__model: ResNet = model
        self.__root: Node = None
        self.__depth_reached: int = 0
        self.__uniform_policy: np.ndarray = np.ones(
            self.__action_space.size) / self.__action_space.size

    def set_root(self, state: chess.Board) -> None:
        """Set the root node of the tree"""

        self.__root = Node(self.__c_puct, state, self.__action_space)

    def calculate_policy_value(self, node: Node) -> np.ndarray:
        """Calculate the policy and value of a given node"""

        if self.__model is None:
            policy, value = self.__uniform_policy, 0
        else:
            with torch.no_grad():
                policy, value = self.__model(
                    self.__model.get_unbatched_tensor_state(node.encoded))
            policy = self.__model.get_policy(policy)
            value = self.__model.get_value(value)
        return policy, value

    def search_step(self, max_depth: int) -> None:
        """Performs one iteration of the MCTS algorithm"""

        node = self.__root
        # select
        depth_reached = 0
        while node.is_expanded and depth_reached < max_depth:
            node = node.select()
            node.visit_count += 1
            depth_reached += 1
        if depth_reached > self.__depth_reached:
            self.__depth_reached = depth_reached
        # apply value of -infinity to the selected node, to prevent other threads from selecting it
        node.apply_virtual_loss()
        if node.is_checkmate:
            value = 1
        elif node.is_terminal:
            value = 0
        else:
            value = None

        # expand if not terminal
        if value is None:
            if node.id in self.__cache:
                policy, value = self.__cache[node.id]
            else:
                policy, value = self.calculate_policy_value(node)
                self.__cache[node.id] = (policy, value)
            node.expand(policy)
        node.revert_virtual_loss()
        node.backpropagate(value)

    def get_dist(self, temperature: float) -> np.ndarray:
        """Returns the visit count distribution of the root node"""

        visit_counts = self.__root.child_visit_count
        if temperature == 0:
            action = np.argmax(visit_counts)
            distribution = np.zeros(self.__action_space.size)
            distribution[action] = 1
            return distribution
        else:
            visit_counts = np.power(visit_counts, 1 / temperature)
            visit_counts = visit_counts / np.sum(visit_counts)
            return visit_counts

    def root_from_fen(self, fen: str) -> None:
        """Sets the root node from a fen string"""

        self.set_root(chess.Board(fen))
        policy, _ = self.calculate_policy_value(self.__root)
        self.__root.expand(policy)

    def restrict_root(self, allowed_actions: list[chess.Move]) -> None:
        """Restricts the root node to the given actions"""

        # remove all nodes that are not in the given actions
        allowed_actions = [self.__action_space.get_key(
            action) for action in allowed_actions]
        for action in self.__root.children:
            if action not in allowed_actions:
                self.__root.child_value[action] = -np.inf

    def select_child_as_root(self, action: chess.Move) -> None:
        """Selects the given action as the root node"""

        action = self.__action_space.get_key(action)
        self.__root = self.__root.children[action]

    def best_move(self) -> chess.Move:
        """Returns the best move and ponder move of the root node"""

        best_action = self.__root.best_child
        ponder_action = self.__root.children[best_action].best_child
        return self.__action_space[best_action], self.__action_space[ponder_action]

    def best_line(self) -> list[chess.Move]:
        """Returns the best line of the root node"""

        line = []
        node = self.__root
        while node.is_expanded:
            node = node.select()
            line.append(self.__action_space[node.action].uci())
        return line

    @property
    def depth(self) -> int:
        """Return the maximum depth reached in the tree"""

        return self.__depth_reached

    @property
    def evaluation(self) -> float:
        """Return the evaluation of the root node as a value between -1 and 1"""

        best_child = self.__root.best_child
        return self.__root.child_value[best_child] / (self.__root.child_visit_count[best_child] + 1)

    def reset_cache(self) -> None:
        """Resets the tree"""

        self.__cache.clear()

    def reset(self) -> None:
        """Resets the tree"""

        self.__root = None
        self.reset_cache()

    @property
    def c_puct(self) -> np.float32:
        """Return the c_puct value"""

        return self.__c_puct

    @c_puct.setter
    def c_puct(self, value: np.float32) -> None:
        """Set the c_puct value"""

        self.__c_puct = value

    @property
    def root(self) -> Node:
        """Return the root node"""

        return self.__root
