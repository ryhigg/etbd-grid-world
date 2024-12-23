import matplotlib.pyplot as plt
import os
from src.algorithms import (
    built_in_reinforcement_decay,
    emergent_reinforcement_decay,
    emergent_reinforcement_decay_all_states,
    built_in_reinforcement_decay_all_states,
)
from src.animator import Animator


def main():
    TRIALS = 250
    AOS = 5
    ALGO = "built_in_all_states"

    for ao in range(0, AOS):
        path = f"figs/{ALGO}/ao{ao}"
        if not os.path.exists(path):
            os.makedirs(path)

        if ALGO == "built_in":
            totals, lengths, paths, final = built_in_reinforcement_decay(TRIALS, 5, 5)
        elif ALGO == "emergent":
            totals, lengths, paths, final = emergent_reinforcement_decay(TRIALS, 5, 5)
        elif ALGO == "emergent_all_states":
            totals, lengths, paths, final = emergent_reinforcement_decay_all_states(
                TRIALS, 5, 5
            )
        elif ALGO == "built_in_all_states":
            totals, lengths, paths, final = built_in_reinforcement_decay_all_states(
                TRIALS, 5, 5
            )
        else:
            raise ValueError("Invalid algorithm")

        trials_to_animate = [0, 1, 2, 9, 99, 249]
        for trial in trials_to_animate:
            print(f"Path for trial {trial}")
            print(paths[trial])
            animator = Animator(5, 5, paths[trial])
            animator.save_animation(f"{path}/trial_{trial}.gif")

        plt.plot(totals)
        plt.xlabel("Trial")
        plt.ylabel("Reward")
        plt.savefig(f"{path}/total_rewards.png")
        plt.close()


if __name__ == "__main__":
    main()
