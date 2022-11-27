from copy import deepcopy
from typing_extensions import Self

# Dimension of the board, here 3x3.
DIMENSION = 3

# Representation of the empty tile.
EMPTY = "0"

# Debug mode flag.
DEBUG = True

# Initial state for debug purpose.
DEBUG_START_STATE = [["2", "1", "3"], [EMPTY, "8", "4"], ["6", "7", "5"]]

# Target state for debug purpose.
DEBUG_TARGET_STATE = [["1", "2", "3"], ["8", EMPTY, "4"], ["7", "6", "5"]]

# Coordinate of a tile, where (0, 0) is the left-upper corner of the board.
Point = tuple[int, int]

# State of the board, represented by a 2-dimensional matrix.
State = list[list[str]]


class Node:
    """
    A* tree node.
    """

    def __init__(self, state: State, target: State, parent: Self | None = None) -> None:
        """
        Initialize a node.

        Args:
            state: The current state.
            target: The target state.
            parent: The parent node of this node, default to None.
        """
        self.__state = state
        self.__target = target
        self.__parent = parent

        self.__g = self.__get_g()
        self.__h = self.__get_h()
        self.__f = self.__g + self.__h

    def __str__(self) -> str:
        """
        Display the current state in a user-friendly way.

        Returns:
            The user-friendly string representation of the current state.

        Example:
            State [["1", "2", "3"], ["4", "5", "6"], ["7", "8", "0"]]
            is displayed as
            1 2 3
            4 5 6
            7 8 0
            .
        """
        return "\n".join(
            [
                " ".join([" " if tile == EMPTY else tile for tile in line])
                for line in self.__state
            ]
        )

    def __repr__(self) -> str:
        """
        Display the current state in a developer-friendly way.

        Returns:
            The developer-friendly string representation of the current state.

        Note:
            This is used as the key of the history map.

        Example:
            State [["1", "2", "3"], ["4", "5", "6"], ["7", "8", "0"]]
            is displayed as 123456780.
        """
        return "".join(["".join(line) for line in self.__state])

    def __lt__(self, next: Self) -> bool:
        """
        Override "<" operator.
        """
        return self.__f < next.__f

    def __gt__(self, next: Self) -> bool:
        """
        Override ">" operator.
        """
        return self.__f > next.__f

    @property
    def f(self):
        return self.__f

    def __find_empty(self) -> Point:
        """
        Find the coordinates of the empty tile.

        Returns:
            The coordinates of the empty tile.

        Throws:
            Exception if the empty tile is not found.
        """
        for line_index, line in enumerate(self.__state):
            for column_index, tile in enumerate(line):
                if tile == EMPTY:
                    return line_index, column_index

        raise Exception("Empty tile is not found")

    def __get_next_empties(self, current_empty: Point) -> list[Point]:
        """
        Find the next coordinates that the empty tile can move to.

        Args:
            empty: The current coordinates of the empty tile.

        Returns:
            A list of coordinates that the empty tile can move to.
        """
        return list(
            filter(
                lambda point: (
                    0 <= point[0] <= DIMENSION - 1 and 0 <= point[1] <= DIMENSION - 1
                ),
                [
                    (current_empty[0] + 1, current_empty[1]),
                    (current_empty[0] - 1, current_empty[1]),
                    (current_empty[0], current_empty[1] + 1),
                    (current_empty[0], current_empty[1] - 1),
                ],
            )
        )

    def __get_next_state(self, current_empty: Point, next_empty: Point) -> State:
        """
        Get the next state after the empty tile has moved.

        Args:
            current_empty: The current coordinates of the empty tile.
            next_empty: The next coordinates of the empty tile.

        Returns:
            The next state after the empty tile is moved.
        """
        next_state = deepcopy(self.__state)
        (
            next_state[next_empty[0]][next_empty[1]],
            next_state[current_empty[0]][current_empty[1]],
        ) = (
            next_state[current_empty[0]][current_empty[1]],
            next_state[next_empty[0]][next_empty[1]],
        )
        return next_state

    def __get_g(self) -> int:
        """
        Get the minimal cost of the current node,
        by counting the depth of the current node.

        Returns:
            The minimal cost of the current node.
        """
        return 0 if self.__parent == None else self.__parent.__g + 1

    def __get_h(self) -> int:
        """
        Get the heuristic value of the current node,
        by counting the number of incorrectly placed tiles.

        Args:
            target: The target state.

        Returns:
            The heuristic value of the current node.
        """
        count = 0
        for i in range(DIMENSION):
            for j in range(DIMENSION):
                if self.__target[i][j] not in (self.__state[i][j], EMPTY):
                    count += 1
        return count

    def create_children(self) -> list[Self]:
        """
        Create the children of this node,
        whose state can be derived from the current state
        by moving one tile towards a certain direction.

        Returns:
            A list of children of this node.
        """
        empty = self.__find_empty()
        next_empties = self.__get_next_empties(empty)
        return [
            Node(self.__get_next_state(empty, next), self.__target, self)
            for next in next_empties
        ]

    def is_target(self) -> bool:
        """
        Check if the current state is the target state,
        by comparing its heuristic value to 0.

        Returns:
            True if the current state is the target state, or False otherwise.
        """
        return self.__h == 0

    def print_branch(self) -> None:
        """
        Print every node in the current branch.
        """
        if self.__parent != None:
            self.__parent.print_branch()
        print(self, end="\n\n")


class CandidateQueue:
    """
    Priority queue of the candidate A* tree nodes,
    where the nodes with the minimal f-value
    are placed at the front of the queue.

    Note:
        Also known as the open list in A* algorithm.
    """

    def __init__(self, nodes: list[Node] = []) -> None:
        """
        Initialize the queue with a list of nodes.

        Args:
            nodes: A list of nodes.
        """
        self.__queue = nodes

    def push(self, node: Node | list[Node]) -> None:
        """
        Push one node or a list of nodes into the queue.

        Args:
            node: The node(s) to be pushed.
        """
        if isinstance(node, list):
            self.__queue.extend(node)
        else:
            self.__queue.append(node)

        self.__queue.sort()

    def pop(self) -> Node:
        """
        Pop a node from the queue.

        Returns:
            The node with the minimal f-value.
        """
        return self.__queue.pop(0)

    def is_empty(self) -> bool:
        """
        Check if the queue is empty.

        Returns:
            True if the queue is empty, or False otherwise.
        """
        return len(self.__queue) == 0


class HistoryMap:
    """
    Hash map of the visited A* nodes,
    where the key is the string representation of the state of the node,
    and the value is the node itself.

    Note:
        Also known as the closed list in A* algorithm.
    """

    def __init__(self) -> None:
        """
        Initialize the map with an empty dictionary.
        """
        self.__map: dict[str, Node] = {}

    def add(self, node: Node) -> None:
        """
        Add a node into the map.

        Args:
            node: The node to be added.
        """
        self.__map[repr(node)] = node

    def contains(self, node: Node) -> bool:
        """
        Check if the map contains a node.

        Args:
            node: The node to be checked.

        Returns:
            True if the map contains the node, or False otherwise.
        """
        return repr(node) in self.__map


def read_state():
    """
    Read a state from user input.
    """
    return [list(input(">>> ").replace(" ", ""))[:DIMENSION] for _ in range(DIMENSION)]


def go(start_state: State, target_state: State) -> int:
    """
    Solve 8 puzzle problem.

    Args:
        start_state: The initial state.
        target_state: The target state.

    Returns:
        The number of steps to reach the target state, or -1 on failure.
    """
    candidates = CandidateQueue([Node(start_state, target_state)])
    history = HistoryMap()

    while not candidates.is_empty():
        candidate = candidates.pop()

        if candidate.is_target():
            candidate.print_branch()
            return candidate.f

        if history.contains(candidate):
            continue

        children = candidate.create_children()
        candidates.push(children)
        history.add(candidate)

    return -1


if __name__ == "__main__":
    print("\nEnter initial state:")
    start_state = DEBUG_START_STATE if DEBUG else read_state()
    print("\nEnter target state:")
    target_state = DEBUG_TARGET_STATE if DEBUG else read_state()
    print()

    steps = go(start_state, target_state)

    if steps == -1:
        print("Fail to find a solution.")
    else:
        print(f"Solved in {steps} step(s).")
