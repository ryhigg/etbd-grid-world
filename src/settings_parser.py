import json

DEFAULTS = {
    "grid_config": {
        "rows": 5,
        "cols": 5,
        "terminal_states": [24],
        "agent_start": 0,
    },
    "organism_config": {
        "lower_phenotype_bound": 0,
        "upper_phenotype_bound": 1024,
        "population_size": 100,
        "fdf_form": "linear",
        "fdf_mean": 40,
        "mutation_rate": 0.1,
    },
    "algorithm_config": {
        "type": "emergent",
        "fdf_mean": 40,
        "trials": 250,
    },
}


def read_file(file_path):
    """
    Read a JSON file and return the data as a dictionary
    """
    if not file_path.endswith(".json"):
        raise ValueError("File must be a JSON file")

    with open(file_path, "r") as file:
        data = json.load(file)

    return data


def load_settings(file_path):
    """
    Load settings from a JSON file and return a dictionary
    """
    data = read_file(file_path)

    settings = DEFAULTS.copy()
    settings.update(data)

    return settings
