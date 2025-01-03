from src.algorithms import EmergentReinforcementDecay
from src.organism import Organism
from src.grid import GridWorld
import pandas as pd

# Group 1: A = B
# Group 2: A > B

# Phase 1: Patch A baited
# Phase 2: Patch B unbaited
# Phase 3: Patch A unbaited
# Phase 4: Patch B baited
# Phase 5: test


data = {
    "Group": [],
    "Delay": [],
    "Phase": [],
    "Path": [],
    "Latency": [],
}

PATCH_A_ON = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    21,
    22,
    23,
    24,
    25,
    26,
    31,
    32,
    33,
    34,
    35,
    41,
    42,
    43,
    44,
    51,
    52,
    53,
    61,
    62,
    71,
]
PATCH_B_ON = [
    9,
    18,
    19,
    27,
    28,
    29,
    36,
    37,
    38,
    39,
    45,
    46,
    47,
    48,
    49,
    54,
    55,
    56,
    57,
    58,
    59,
    63,
    64,
    65,
    66,
    67,
    68,
    69,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
]

DELAYS = [0, 5, 10, 20, 40, 80, 160, 320]
PHASES = 5
TRIALS = 8


for group in range(1, 3):
    for delay in DELAYS:
        grid = GridWorld(
            {
                "rows": 9,
                "cols": 9,
                "terminal_states": [72, 8],
                "unavailable_states": [],
                "agent_start": 40,
            }
        )
        org = Organism(
            grid,
            {
                "lower_phenotype_bound": 0,
                "upper_phenotype_bound": 1024,
                "population_size": 100,
                "fdf_form": "linear",
                "mutation_rate": 0.1,
            },
        )
        algo = EmergentReinforcementDecay(grid, org, {"trials": TRIALS, "fdf_mean": 40})
        for phase in range(PHASES):

            if phase == 0:
                grid.set_available_states(PATCH_A_ON)
                if group == 2:
                    algo.fdf_mean = 20

            elif phase == 1:
                grid.set_available_states(PATCH_B_ON)
                algo.fdf_mean = 5000

            elif phase == 2:
                grid.set_available_states(PATCH_A_ON)
                algo.fdf_mean = 5000

            elif phase == 3:
                grid.set_available_states(PATCH_B_ON)
                if group == 2:
                    algo.fdf_mean = 100

            elif phase == 4:
                grid.set_available_states([])
                algo.fdf_mean = 5000
                TRIALS = 1
                for val in range(delay):
                    for state in range(81):
                        org.no_reinforcer_delivered(state)

            algo.trials = TRIALS

            algo.run()
            total_rewards, path_lengths, paths = algo.get_output()

            data["Group"].append(group)
            data["Delay"].append(delay)
            data["Phase"].append(phase)
            data["Path"].append(paths)
            data["Latency"].append(path_lengths)

            print(
                f"Group: {group}, Delay: {delay}, Phase: {phase}, Path: {paths}, Latency: {path_lengths}"
            )

df = pd.DataFrame(data)
df.to_csv("data.csv", index=False)
