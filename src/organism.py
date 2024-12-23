import numpy as np
from pyetbd.rules import fdfs, fitness_calculation, selection, recombination, mutation
from src.grid import GridWorld


class Organism:
    def __init__(self, grid: GridWorld, config):
        self.grid = grid
        self.config = config

        self.lower_phenotype_bound = config["lower_phenotype_bound"]
        self.upper_phenotype_bound = config["upper_phenotype_bound"]
        self.population_size = config["population_size"]
        self.fdf_form = config["fdf_form"]
        self.mutation_rate = config["mutation_rate"]
        self.bit_string_length = len(bin(self.upper_phenotype_bound)) - 2

        self.init_populations()
        self.set_fdf(self.fdf_form)

        self.set_sd(grid.agent_position)

    def init_populations(self):
        self.populations = [
            np.random.randint(
                self.lower_phenotype_bound,
                self.upper_phenotype_bound,
                self.population_size,
            )
            for _ in range(len(self.grid.possible_states))
        ]

    def set_sd(self, position):
        self.current_population = self.populations[position]

    def set_fdf(self, fdf_form):
        if fdf_form == "linear":
            self.fdf = fdfs.sample_linear_fdf
        elif fdf_form == "exponential":
            self.fdf = fdfs.sample_exponential_fdf
        else:
            raise ValueError("Invalid FDF form")

    def emit(self):
        return np.random.choice(self.current_population)

    def reinforcer_delivered(self, state, emitted, fdf_mean):
        self.set_sd(state)

        fitness_values = fitness_calculation.get_circular_fitness_values(
            self.current_population, emitted, self.upper_phenotype_bound
        )

        parents = selection.fitness_search_selection(
            self.current_population,
            fitness_values,
            fdf_mean,
            self.fdf,
        )

        offspring = recombination.recombine_parents(
            parents, self.bit_string_length, recombination.bitwise_combine
        )
        mutated_offspring = mutation.bit_flip_mutate(offspring, self.mutation_rate)

        self.populations[state] = mutated_offspring

    def no_reinforcer_delivered(self, state):
        self.set_sd(state)

        parents = selection.randomly_select_parents(self.current_population)
        offspring = recombination.recombine_parents(
            parents, self.bit_string_length, recombination.bitwise_combine
        )
        mutated_offspring = mutation.bit_flip_mutate(offspring, self.mutation_rate)

        self.populations[state] = mutated_offspring
