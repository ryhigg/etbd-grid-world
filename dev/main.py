from src.grid import GridWorld
from src.organism import Organism
import numpy as np
import matplotlib.pyplot as plt
import textwrap

TRIALS = 250


def hyperbolic_decay_fn(x, A, k):
    return (-A / (1 + k * x)) + A


def emergent_reinforcement_decay():
    grid = GridWorld(5, 5)
    organism = Organism(grid)
    total_rewards = np.zeros(TRIALS)
    path_lengths = np.zeros(TRIALS)

    for trial in range(TRIALS):
        states = []
        emissions = []
        trial_reward = 0

        grid.reset()
        current_state = grid.agent_position
        organism.set_sd(current_state)

        done = False
        while not done:
            organism.set_sd(current_state)
            emitted = organism.emit()

            action = grid.get_action(emitted)

            if action is None:
                trial_reward -= 1
                continue

            states.insert(0, current_state)
            emissions.insert(0, emitted)

            organism.no_reinforcer_delivered(current_state)

            current_state, reward, done, _ = grid.step(action)
            trial_reward += reward

            if done:
                print(
                    textwrap.dedent(
                        f"""
                ---------------------
                Trial {trial} finished with reward {trial_reward}
                Path length: {len(states)}
                Path states: {states}
                ---------------------
                """
                    )
                )
                total_rewards[trial] = trial_reward
                path_lengths[trial] = len(states)
                states_updated = []

                path_length = len(states)
                for i in range(path_length):
                    if states[i] in states_updated:
                        continue

                    states_updated.append(states[i])

                    organism.reinforcer_delivered(states[i], emissions[i])
                    # we do a no reinforcer cycle for the number of steps back in the path
                    for j in range(i):
                        organism.no_reinforcer_delivered(states[i])

        if trial == TRIALS - 1:
            print("Final path:")
            print(states[::-1])

            for state in states:
                grid.grid[state // grid.rows, state % grid.cols] = 1

            grid.render()

    print("Emergent Reinforcement Decay")
    print(f"Average reward: {np.mean(total_rewards)}")
    print(f"Average path length: {np.mean(path_lengths)}")
    fig, ax = plt.subplots()
    ax.plot(total_rewards)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Reward")
    ax.set_title("Emergent Reinforcement Decay")
    plt.show()


def built_in_reinforcement_decay():
    grid = GridWorld(5, 5)
    organism = Organism(grid)
    total_rewards = np.zeros(TRIALS)
    path_lengths = np.zeros(TRIALS)

    for trial in range(TRIALS):
        states = []
        emissions = []
        trial_reward = 0

        grid.reset()
        current_state = grid.agent_position
        organism.set_sd(current_state)

        done = False
        while not done:
            organism.set_sd(current_state)
            emitted = organism.emit()

            action = grid.get_action(emitted)

            if action is None:
                trial_reward -= 1
                continue

            states.insert(0, current_state)
            emissions.insert(0, emitted)

            organism.no_reinforcer_delivered(current_state)

            current_state, reward, done, _ = grid.step(action)
            trial_reward += reward

            if done:
                print(
                    textwrap.dedent(
                        f"""
                ---------------------
                Trial {trial} finished with reward {trial_reward}
                Path length: {len(states)}
                Path states: {states}
                ---------------------
                """
                    )
                )
                total_rewards[trial] = trial_reward
                path_lengths[trial] = len(states)
                states_updated = []

                path_length = len(states)
                for i in range(path_length):
                    if states[i] in states_updated:
                        continue

                    fdf_mean = hyperbolic_decay_fn(i, 300, 0.008)

                    states_updated.append(states[i])

                    organism.reinforcer_delivered(
                        states[i], emissions[i], fdf_mean=fdf_mean
                    )

        if trial == TRIALS - 1:
            print("Final path:")
            print(states[::-1])

            for state in states:
                grid.grid[state // grid.rows, state % grid.cols] = 1

            grid.render()
    print("Built-in Reinforcement Decay")
    print(f"Average reward: {np.mean(total_rewards)}")
    print(f"Average path length: {np.mean(path_lengths)}")
    fig, ax = plt.subplots()
    ax.plot(total_rewards)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Reward")
    ax.set_title("Built-in Reinforcement Decay")
    plt.show()


if __name__ == "__main__":
    built_in_reinforcement_decay()
    # emergent_reinforcement_decay()
