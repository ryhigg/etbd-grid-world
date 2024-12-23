import numpy as np


class GridWorld:
    def __init__(self, config: dict):
        self.config = config
        self.rows = config["rows"]
        self.cols = config["cols"]
        self.terminal_states = config["terminal_states"]
        self.agent_position = config["agent_start"]
        self.grid = np.zeros((self.rows, self.cols))

        self.possible_states = [i for i in range(self.rows * self.cols)]
        self.actions_dict = {"U": -self.rows, "D": self.rows, "L": -1, "R": 1}
        self.actions_list = ["U", "D", "L", "R"]

    def agent_in_terminal_state(self, state) -> tuple[bool, int]:
        """
        returns True if the agent is in a terminal state, False otherwise
        also returns the state
        """

        return state in self.terminal_states, state

    def get_agent_position(self) -> tuple[int, int]:
        """
        returns the agent's position in the grid
        """

        x, y = self.agent_position // self.rows, self.agent_position % self.cols
        return x, y

    def set_agent_position(self, state: int):
        """
        sets the agent's position in the grid
        """

        x, y = self.get_agent_position()
        self.grid[x, y] = 0
        self.agent_position = state
        x, y = self.get_agent_position()
        self.grid[x, y] = 1

    def off_grid_move(self, new_state: int, old_state: int) -> bool:
        """
        returns True if the agent is trying to move off the grid, False otherwise
        """

        if new_state not in self.possible_states:
            return True

        elif old_state % self.rows == 0 and new_state % self.rows == self.rows - 1:
            return True

        elif old_state % self.rows == self.rows - 1 and new_state % self.rows == 0:
            return True

        else:
            return False

    def step(self, action: str) -> tuple[int, int, bool, None]:
        """
        takes an action and returns the new state, reward, done, and info
        """

        x, y = self.get_agent_position()
        new_state = self.agent_position + self.actions_dict[action]

        if self.off_grid_move(new_state, self.agent_position):
            new_state = self.agent_position

        done, new_state = self.agent_in_terminal_state(new_state)
        reward = -1 if not done else 0
        self.set_agent_position(new_state)

        return new_state, reward, done, None

    def reset(self) -> int:
        """
        resets the agent to the starting position
        """

        self.agent_position = self.config["agent_start"]
        self.grid = np.zeros((self.rows, self.cols))
        return self.agent_position

    def get_action(self, emitted: int):
        """
        returns the action based on the emitted signal
        """

        if 448 <= emitted <= 511:
            return "U"
        elif 512 <= emitted <= 575:
            return "D"
        elif 128 <= emitted <= 191:
            return "L"
        elif 832 <= emitted <= 895:
            return "R"
        else:
            return None

    def render(self):
        """
        prints the grid
        """

        print("".join(["-------" for _ in range(self.cols)]))
        for row in self.grid:
            for cell in row:
                if cell == 0:
                    print("-", end="\t")
                elif cell == 1:
                    print("X", end="\t")
                elif cell % 2 == 0:
                    print(f"{int(cell)}N", end="\t")
                else:
                    print(f"{int(cell)-1}O", end="\t")
            print("\n")
        print("".join(["-------" for _ in range(self.cols)]))


class GridWorld1:
    def __init__(self, rows, cols, portals={}):
        self.grid = np.zeros((rows, cols))
        self.rows = rows
        self.cols = cols

        self.possible_states = [i for i in range(self.rows * self.cols - 1)]
        self.possible_states_plus_terminal = [i for i in range(self.rows * self.cols)]
        self.actions_dict = {"U": -self.rows, "D": self.rows, "L": -1, "R": 1}
        self.actions_list = ["U", "D", "L", "R"]

        self.add_portals(portals)
        self.agent_position = 0

    def add_portals(self, portals):
        self.portals = portals

        i = 2
        for entrance, exit in portals.items():
            x, y = entrance // self.rows, entrance % self.cols
            self.grid[x, y] = i
            i += 1
            x, y = exit // self.rows, exit % self.cols
            self.grid[x, y] = i
            i += 1

    def agent_in_terminal_state(self, state):
        return (
            state in self.possible_states_plus_terminal
            and state not in self.possible_states
        )

    def get_agent_position(self):
        x, y = self.agent_position // self.rows, self.agent_position % self.cols
        return x, y

    def set_agent_position(self, state):
        x, y = self.get_agent_position()
        self.grid[x, y] = 0
        self.agent_position = state
        x, y = self.get_agent_position()
        self.grid[x, y] = 1

    def off_grid_move(self, new_state, old_state):
        if new_state not in self.possible_states_plus_terminal:
            return True

        elif old_state % self.rows == 0 and new_state % self.rows == self.rows - 1:
            return True

        elif old_state % self.rows == self.rows - 1 and new_state % self.rows == 0:
            return True

        else:
            return False

    def step(self, action):
        x, y = self.get_agent_position()
        new_state = self.agent_position + self.actions_dict[action]

        if new_state in self.portals.keys():
            new_state = self.portals[new_state]

        if self.off_grid_move(new_state, self.agent_position):
            new_state = self.agent_position

        reward = -1 if not self.agent_in_terminal_state(new_state) else 0
        self.set_agent_position(new_state)

        return (
            new_state,
            reward,
            self.agent_in_terminal_state(new_state),
            None,
        )

    def reset(self):
        self.agent_position = 0
        self.grid = np.zeros((self.rows, self.cols))
        self.add_portals(self.portals)
        return self.agent_position

    def get_action(self, emitted):
        if 448 <= emitted <= 511:
            return "U"
        elif 512 <= emitted <= 575:
            return "D"
        elif 128 <= emitted <= 191:
            return "L"
        elif 832 <= emitted <= 895:
            return "R"
        else:
            return None

    def render(self):
        print("".join(["-------" for _ in range(self.cols)]))
        for row in self.grid:
            for cell in row:
                if cell == 0:
                    print("-", end="\t")
                elif cell == 1:
                    print("X", end="\t")
                elif cell % 2 == 0:
                    print(f"{int(cell)}N", end="\t")
                else:
                    print(f"{int(cell)-1}O", end="\t")
            print("\n")
        print("".join(["-------" for _ in range(self.cols)]))
