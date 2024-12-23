import numpy as np
from pyetbd.rules import fdfs, fitness_calculation, selection, recombination, mutation
from src.grid import GridWorld


class Organism:
    def __init__(self, grid: GridWorld):
        self.grid = grid

        self.populations = [
            np.random.randint(0, 1024, 100) for _ in range(len(grid.possible_states))
        ]
        self.set_sd(grid.agent_position)

    def set_sd(self, position):
        self.current_population = self.populations[position]

    def emit(self):
        return np.random.choice(self.current_population)

    def reinforcer_delivered(self, state, emitted, fdf_mean=40.0):
        self.set_sd(state)

        fitness_values = fitness_calculation.get_circular_fitness_values(
            self.current_population, emitted, 1024
        )
        # parents = selection.fitness_search_selection(
        #     self.current_population,
        #     fitness_values,
        #     fdf_mean,
        #     fdfs.sample_linear_fdf,
        # )
        parents = selection.fitness_search_selection(
            self.current_population,
            fitness_values,
            fdf_mean,
            fdfs.sample_exponential_fdf,
        )
        offspring = recombination.recombine_parents(
            parents, 10, recombination.bitwise_combine
        )
        mutated_offspring = mutation.bit_flip_mutate(offspring, 0.1)

        self.populations[state] = mutated_offspring

    def no_reinforcer_delivered(self, state):
        self.set_sd(state)

        parents = selection.randomly_select_parents(self.current_population)
        offspring = recombination.recombine_parents(
            parents, 10, recombination.bitwise_combine
        )
        mutated_offspring = mutation.bit_flip_mutate(offspring, 0.1)

        self.populations[state] = mutated_offspring
