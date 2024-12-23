import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Animator:
    def __init__(self, rows, cols, path):
        """
        Initialize the animator

        :param rows: Number of rows in the grid
        :param cols: Number of columns in the grid
        :param path: List of states representing the organism's movement
        """
        self.rows = rows
        self.cols = cols
        self.path = path
        self.grid = np.zeros((rows, cols))

        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=(8, 8))

        # Set up grid visualization
        self.ax.set_xlim(-0.5, cols - 0.5)
        self.ax.set_ylim(-0.5, rows - 0.5)
        self.ax.set_xticks(range(cols))
        self.ax.set_yticks(range(rows))
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.grid(True, linestyle="--", color="lightgray")

        # Setup for scatter plot of current position
        self.scatter = self.ax.scatter([], [], color="orange", s=200)

        # Setup for path visualization
        self.path_line = self.ax.plot([], [], color="blue", alpha=0.5, linewidth=2)[0]

        # Title and labels
        self.ax.set_title("Organism Movement in Grid World")

    def state_to_position(self, state):
        """
        Convert a state number to (row, col) position

        :param state: State number in the grid
        :return: Tuple of (row, col)
        """
        return state // self.cols, state % self.cols

    def update_frame(self, i):
        """
        Update animation frame

        :param i: Current frame index
        :return: Tuple of scatter and path line objects
        """
        if i < len(self.path):
            # Get current state and convert to position
            current_state = self.path[i]
            row, col = self.state_to_position(current_state)

            # Update current position (scatter plot)
            self.scatter.set_offsets([(col, row)])

            # Update path
            path_positions = [
                self.state_to_position(state) for state in self.path[: i + 1]
            ]
            path_cols = [pos[1] for pos in path_positions]
            path_rows = [pos[0] for pos in path_positions]

            self.path_line.set_data(path_cols, path_rows)

        return self.scatter, self.path_line

    def init_animation(self):
        """
        Initialize the animation

        :return: Tuple of scatter and path line objects
        """
        self.scatter.set_offsets(np.array([[], []]).T)
        self.path_line.set_data([], [])
        return self.scatter, self.path_line

    def create_animation(self, interval=400):
        """
        Create the animation

        :param interval: Time between frames in milliseconds
        :return: Animation object
        """
        if not self.path:
            raise ValueError("No path provided for animation")

        anim = animation.FuncAnimation(
            self.fig,
            self.update_frame,
            init_func=self.init_animation,
            frames=len(self.path),
            interval=interval,
            blit=True,
            repeat_delay=2000,
        )

        return anim

    def save_animation(self, filename="organism_movement.gif", writer="pillow"):
        """
        Save the animation to a file

        :param filename: Output filename
        :param writer: Animation writer (default is pillow for gif)
        """
        anim = self.create_animation()
        anim.save(filename, writer=writer)
        plt.close(self.fig)

    def show_animation(self):
        """
        Display the animation
        """
        anim = self.create_animation()
        plt.show()
