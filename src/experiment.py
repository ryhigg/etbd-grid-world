from src.grid import GridWorld
from src.organism import Organism
from src.algorithms import (
    EmergentReinforcementDecay,
    BuiltInReinforcementDecay,
    BuiltInReinforcementDecayAllStates,
    EmergentReinforcementDecayAllStates,
)
from src.animator import Animator
import numpy as np
import matplotlib.pyplot as plt


class Experiment:
    algo_map = {
        "built-in_all_states": BuiltInReinforcementDecayAllStates,
        "emergent_all_states": EmergentReinforcementDecayAllStates,
        "built-in": BuiltInReinforcementDecay,
        "emergent": EmergentReinforcementDecay,
    }

    def __init__(self, config):
        self.config = config

        self.grid = GridWorld(config["grid_config"])
        self.organism = Organism(self.grid, config["organism_config"])
        self.algorithm = self.algo_map[config["algorithm_config"]["type"]](
            self.grid, self.organism, config["algorithm_config"]
        )

    def run(self, plot=True, animate=True):
        self.algorithm.run()
        if plot:
            self.plot_results()
        if animate:
            self.animate()

        print("Experiment Statistics:")
        print("Algorithm:", self.config["algorithm_config"]["type"])
        print("Mean Total Reward:", np.mean(self.algorithm.total_rewards))
        print("Mean Path Length:", np.mean(self.algorithm.path_lengths))
        print("Final Path:", self.algorithm.paths[-1])

    def plot_results(self):
        total_rewards, path_lengths, paths = self.algorithm.get_output()
        fig, ax = plt.subplots()
        ax.plot(total_rewards)
        ax.set_xlabel("Trials")
        ax.set_ylabel("Total Reward")
        plt.show()

    def animate(self):
        total_rewards, path_lengths, paths = self.algorithm.get_output()
        final_path = paths[-1]
        animator = Animator(self.grid.rows, self.grid.cols, final_path)
        animator.show_animation()
