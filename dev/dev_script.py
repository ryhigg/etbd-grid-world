import numpy as np
from pyetbd.rules import fdfs, fitness_calculation, selection, recombination, mutation
import matplotlib.pyplot as plt


class GridWorld(object):
    def __init__(self, m, n, magicSquares={}):
        self.grid = np.zeros((m, n))
        self.m = m
        self.n = n

        self.stateSpace = [i for i in range(self.m * self.n)]
        self.stateSpace.remove(self.m * self.n - 1)
        self.stateSpacePlus = [i for i in range(self.m * self.n)]
        self.actionSpace = {"U": -self.m, "D": self.m, "L": -1, "R": 1}
        self.possibleActions = ["U", "D", "L", "R"]
        self.addMagicSquares(magicSquares)
        self.agentPosition = 0

    def addMagicSquares(self, magicSquares):
        self.magicSquares = magicSquares

        i = 2
        for square in magicSquares:
            x = square // self.m
            y = square % self.n
            self.grid[x, y] = i
            i += 1
            x = magicSquares[square] // self.m
            y = magicSquares[square] % self.n
            self.grid[x, y] = i
            i += 1

    def isTerminalState(self, state):
        return state in self.stateSpacePlus and state not in self.stateSpace

    def getAgentRowAndColumn(self):
        x = self.agentPosition // self.m
        y = self.agentPosition % self.n
        return x, y

    def setState(self, state):
        x, y = self.getAgentRowAndColumn()
        self.grid[x][y] = 0
        self.agentPosition = state
        self.grid[x][y] = 1

    def offGridMove(self, newState, oldState):
        if newState not in self.stateSpacePlus:
            return True

        elif oldState % self.m == 0 and newState % self.m == self.m - 1:
            return True

        elif oldState % self.m == self.m - 1 and newState % self.m == 0:
            return True

        else:
            return False

    def step(self, action):
        x, y = self.getAgentRowAndColumn()
        resultingState = self.agentPosition + self.actionSpace[action]
        if resultingState in self.magicSquares.keys():
            resultingState = self.magicSquares[resultingState]

        reward = -1 if not self.isTerminalState(resultingState) else 0
        if not self.offGridMove(resultingState, self.agentPosition):
            self.setState(resultingState)
            return resultingState, reward, self.isTerminalState(resultingState), None
        else:
            return (
                self.agentPosition,
                reward,
                self.isTerminalState(self.agentPosition),
                None,
            )

    def reset(self):
        self.agentPosition = 0
        self.grid = np.zeros((self.m, self.n))
        self.addMagicSquares(self.magicSquares)
        return self.agentPosition

    def render(self):
        buffer = ["-" for _ in range(self.n * 7)]
        print("".join(buffer))
        for row in self.grid:
            for col in row:
                if col == 0:
                    print("-", end="\t")
                elif col == 1:
                    print("X", end="\t")
                elif col == 2:
                    print("Ain", end="\t")
                elif col == 3:
                    print("Aout", end="\t")
                elif col == 4:
                    print("Bin", end="\t")
                elif col == 5:
                    print("Bout", end="\t")
            print("\n")
        print("".join(buffer))

    def acitonSpaceSample(self):
        return np.random.choice(self.possibleActions)

    def getActionFromEmitted(self, emitted):
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


def main():
    grid = GridWorld(5, 5)

    # each possible state on the grid has an associated population of potential behaviors
    populations = [np.random.randint(0, 1024, 100) for _ in range(len(grid.stateSpace))]

    gens = 250
    total_rewards = np.zeros(gens)
    final_path = []

    for gen in range(gens):

        print("Trial: ", gen)

        done = False
        gen_rewards = 0
        current_state = grid.reset()
        gen_states = []
        gen_emissions = []

        while not done:
            current_population = populations[current_state]
            emitted = np.random.choice(current_population)

            action = grid.getActionFromEmitted(emitted)

            if action is None:
                gen_rewards -= 1
                continue

            # if there is an action, we need to record the state and emission
            gen_states.append(current_state)
            gen_emissions.append(emitted)

            # run no reinforcement algo on population
            parents = selection.randomly_select_parents(current_population)
            offspring = recombination.recombine_parents(
                parents, 10, recombination.bitwise_combine
            )
            mutated_offspring = mutation.bit_flip_mutate(offspring, 0.1)
            populations[current_state] = mutated_offspring

            # take the action
            current_state, reward, done, _ = grid.step(action)
            gen_rewards += reward

            # if the agent finds the food, we need to reinforce the behaviors that led to that state
            if done:
                print(f"Found the food! Generation: {gen}, Reward: {gen_rewards}")
                states_updated = []
                for i in range(len(gen_states)):
                    fdf_mean = (
                        -300 / (1 + 0.043 * ((len(gen_states) - i) + 1.225))
                    ) + 300
                    # fdf_mean = 40

                    if gen_states[i] not in states_updated:
                        states_updated.append(gen_states[i])

                        population = populations[gen_states[i]]
                        emitted = gen_emissions[i]

                        fitness_values = (
                            fitness_calculation.get_circular_fitness_values(
                                population, emitted, 1024
                            )
                        )
                        parents = selection.fitness_search_selection(
                            population,
                            fitness_values,
                            fdf_mean,
                            fdfs.sample_linear_fdf,
                        )
                        offspring = recombination.recombine_parents(
                            parents, 10, recombination.bitwise_combine
                        )
                        mutated_offspring = mutation.bit_flip_mutate(offspring, 0.1)

                        populations[gen_states[i]] = mutated_offspring

                        # for j in range(len(gen_states) - i):
                        #     population = populations[gen_states[i]]

                        #     parents = selection.randomly_select_parents(population)
                        #     offspring = recombination.recombine_parents(
                        #         parents, 10, recombination.bitwise_combine
                        #     )
                        #     mutated_offspring = mutation.bit_flip_mutate(offspring, 0.1)

                        #     populations[gen_states[i]] = mutated_offspring

                final_path = gen_states

        total_rewards[gen] = gen_rewards

    for state in final_path:
        grid.grid[state // grid.m][state % grid.n] = 1

    print(f"Final Path: {final_path}")
    grid.render()

    fig, ax = plt.subplots()
    ax.plot(total_rewards)
    ax.set(xlabel="Trial", ylabel="Total Reward", title="Total Reward per Trial")

    plt.show()


if __name__ == "__main__":
    main()
