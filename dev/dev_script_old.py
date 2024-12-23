import numpy as np
from pyetbd.rules import fdfs, fitness_calculation, selection, recombination, mutation
import pandas as pd


class Organism:
    def __init__(self, grid):
        self.grid = grid
        self.populations = [
            [np.random.randint(0, 1024, 100) for _ in range(grid.cols)]
            for _ in range(grid.rows)
        ]

        self.position = (0, 0)  # type: tuple[int, int]

        self.current_population = self.populations[self.position[1]][self.position[0]]

    def emit(self):
        self.emitted = np.random.choice(self.current_population)

    def reinforcer_delivered(self, position, emitted, magnitude):
        population = self.populations[position[1]][position[0]]
        fitness_values = fitness_calculation.get_circular_fitness_values(
            population, emitted, 1024
        )
        parents = selection.fitness_search_selection(
            population, fitness_values, magnitude, fdfs.sample_exponential_fdf
        )
        offspring = recombination.recombine_parents(
            parents, 10, recombination.bitwise_combine
        )
        mutated_offspring = mutation.bit_flip_mutate(offspring, 0.1)

        self.populations[position[1]][position[0]] = mutated_offspring


class Grid:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.grid = [[0 for _ in range(cols)] for _ in range(rows)]

    def print_grid(self):
        for row in self.grid:
            print(row)

    def place_food(self, x, y):
        self.grid[y][x] = 1

    def move_food(self, x, y, new_x, new_y):
        self.grid[y][x] = 0
        self.grid[new_y][new_x] = 1

    def place_org(self, x, y):
        self.grid[y][x] = 2

    def move_org(self, x, y, new_x, new_y):
        self.grid[y][x] = 0
        self.grid[new_y][new_x] = 2

    def reset(self):
        self.grid = [[0 for _ in range(self.cols)] for _ in range(self.rows)]


def main():
    info_dict = {
        "Trial": [],
        "Step": [],
        "Position": [],
        "Emitted": [],
    }

    grid = Grid(5, 5)
    org = Organism(grid)
    org.position = (2, 2)

    for trial in range(10):
        grid.reset()
        org.position = (2, 2)
        grid.place_org(*org.position)
        grid.place_food(1, 3)
        org.current_population = org.populations[org.position[1]][org.position[0]]

        path_taken = []
        times_moved = 0

        for gen in range(1000):
            org.emit()
            current_position = org.position

            if 448 <= org.emitted <= 511:  # move up
                if current_position[1] == 0:
                    continue

                new_position = (current_position[0], current_position[1] - 1)
                org.position = new_position

            elif 512 <= org.emitted <= 575:  # move down
                if current_position[1] == grid.rows - 1:
                    continue

                new_position = (current_position[0], current_position[1] + 1)
                org.position = new_position

            elif 128 <= org.emitted <= 191:  # move left
                if current_position[0] == 0:
                    continue

                new_position = (current_position[0] - 1, current_position[1])
                org.position = new_position

            elif 832 <= org.emitted <= 895:  # move right
                if current_position[0] == grid.cols - 1:
                    continue

                new_position = (current_position[0] + 1, current_position[1])
                org.position = new_position

            if org.position == current_position:
                # If the organism did not move, skip the rest of the loop
                continue

            times_moved += 1

            # we use the current position here because that's the population that was used to emit
            path_taken.append((current_position, org.emitted))

            info_dict["Trial"].append(trial)
            info_dict["Step"].append(times_moved)
            info_dict["Position"].append(current_position)
            info_dict["Emitted"].append(org.emitted)

            if grid.grid[org.position[1]][org.position[0]] == 1:
                print(f"Food located at {org.position} in trial {trial}")
                for position, emitted in path_taken:
                    org.reinforcer_delivered(position, emitted, 40)

                # move to the next trial once the food has been found
                break

            grid.move_org(*current_position, *org.position)

        print(f"Trial {trial} took {times_moved} steps")

    df = pd.DataFrame(info_dict)
    df.to_csv("output.csv", index=False)


if __name__ == "__main__":
    main()
