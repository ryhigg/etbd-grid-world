import numpy as np
from src.grid import GridWorld
from src.organism import Organism


def emergent_reinforcement_decay(trials, rows, cols):
    grid = GridWorld(rows, cols)
    organism = Organism(grid)
    total_rewards = np.zeros(trials)
    path_lengths = np.zeros(trials)
    paths = []
    final_path = []

    for trial in range(trials):
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
            organism.no_reinforcer_delivered(current_state)

            action = grid.get_action(emitted)

            if action is None:
                trial_reward -= 1
                continue

            states.insert(0, current_state)
            emissions.insert(0, emitted)

            # organism.no_reinforcer_delivered(current_state)

            current_state, reward, done, _ = grid.step(action)
            trial_reward += reward

            if done:
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

                states.insert(0, current_state)
                paths.append(states)

        if trial == trials - 1:
            final_path = states[::-1]

    return total_rewards, path_lengths, paths, final_path


def built_in_reinforcement_decay(trials, rows, cols):
    hyperbolic_decay_fn = lambda x, A, k: (-A / (1 + k * x)) + A
    grid = GridWorld(rows, cols)
    organism = Organism(grid)
    total_rewards = np.zeros(trials)
    path_lengths = np.zeros(trials)
    paths = []
    final_path = []

    for trial in range(trials):
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
            organism.no_reinforcer_delivered(current_state)

            action = grid.get_action(emitted)

            if action is None:
                trial_reward -= 1
                continue

            states.insert(0, current_state)
            emissions.insert(0, emitted)

            # organism.no_reinforcer_delivered(current_state)

            current_state, reward, done, _ = grid.step(action)
            trial_reward += reward

            if done:
                total_rewards[trial] = trial_reward
                path_lengths[trial] = len(states)
                states_updated = []

                path_length = len(states)
                for i in range(path_length):
                    if states[i] in states_updated:
                        continue

                    states_updated.append(states[i])

                    fdf_mean = hyperbolic_decay_fn(i, 300, 0.008)

                    organism.reinforcer_delivered(states[i], emissions[i], fdf_mean)

                states.insert(0, current_state)
                paths.append(states)

        if trial == trials - 1:
            final_path = states[::-1]

    return total_rewards, path_lengths, paths, final_path


def emergent_reinforcement_decay_all_states(trials, rows, cols):
    """
    Notes:
    - This still appears to work well, though not quite as well as when each state is only updated for the most recent emission.
    - This code also has the caveat that it must use an FDF that permits any behavior to have some probability of being chosen as a parent (e.g. exponential FDF).
        - If this is not done, in some cases there will be no eligible parents since the population will be updated based on one emission, but the emitted behavior is no longer in the population since it was updated for a previous emission.
    - The organism still learns to find optimal paths, but it takes longer.

    """
    grid = GridWorld(rows, cols)
    organism = Organism(grid)
    total_rewards = np.zeros(trials)
    path_lengths = np.zeros(trials)
    paths = []
    final_path = []

    for trial in range(trials):
        states = []
        emissions = []
        trial_reward = 0

        grid.reset()
        current_state = grid.agent_position
        organism.set_sd(current_state)

        iteration = 0
        done = False
        while not done:
            iteration += 1
            organism.set_sd(current_state)
            emitted = organism.emit()
            organism.no_reinforcer_delivered(current_state)

            action = grid.get_action(emitted)

            if action is None:
                trial_reward -= 1
                continue

            # states.insert(0, current_state)
            # emissions.insert(0, emitted)

            states.append(current_state)
            emissions.append(emitted)

            # organism.no_reinforcer_delivered(current_state)

            current_state, reward, done, _ = grid.step(action)
            trial_reward += reward

            print(f"Trial: {trial}, Iteration: {iteration}, State: {current_state}")

            if done:
                total_rewards[trial] = trial_reward
                path_lengths[trial] = len(states)
                states_updated = []

                path_length = len(states)
                for i in range(path_length):
                    # if states[i] in states_updated:
                    #     continue

                    states_updated.append(states[i])

                    organism.reinforcer_delivered(states[i], emissions[i])
                    # we do a no reinforcer cycle for the number of steps back in the path
                    for j in range(path_length - i):
                        organism.no_reinforcer_delivered(states[i])

                states.append(current_state)
                paths.append(states)

        if trial == trials - 1:
            final_path = states

    return total_rewards, path_lengths, paths, final_path


def built_in_reinforcement_decay_all_states(trials, rows, cols):
    hyperbolic_decay_fn = lambda x, A, k: (-A / (1 + k * x)) + A
    grid = GridWorld(rows, cols)
    organism = Organism(grid)
    total_rewards = np.zeros(trials)
    path_lengths = np.zeros(trials)
    paths = []
    final_path = []

    for trial in range(trials):
        states = []
        emissions = []
        trial_reward = 0

        grid.reset()
        current_state = grid.agent_position
        organism.set_sd(current_state)

        iteration = 0
        done = False
        while not done:
            iteration += 1
            organism.set_sd(current_state)
            emitted = organism.emit()
            organism.no_reinforcer_delivered(current_state)

            action = grid.get_action(emitted)

            if action is None:
                trial_reward -= 1
                continue

            # states.insert(0, current_state)
            # emissions.insert(0, emitted)

            states.append(current_state)
            emissions.append(emitted)

            # organism.no_reinforcer_delivered(current_state)

            current_state, reward, done, _ = grid.step(action)
            trial_reward += reward

            print(f"Trial: {trial}, Iteration: {iteration}, State: {current_state}")

            if done:
                total_rewards[trial] = trial_reward
                path_lengths[trial] = len(states)
                states_updated = []

                path_length = len(states)
                for i in range(path_length):
                    # if states[i] in states_updated:
                    #     continue

                    states_updated.append(states[i])

                    fdf_mean = hyperbolic_decay_fn(path_length - i, 300, 0.008)

                    organism.reinforcer_delivered(states[i], emissions[i], fdf_mean)

                states.append(current_state)
                paths.append(states)

        if trial == trials - 1:
            final_path = states

    return total_rewards, path_lengths, paths, final_path
