from src.experiment import Experiment
from src.settings_parser import load_settings


class Runner:
    def __init__(self, settings_file):
        self.settings = load_settings(settings_file)
        self.experiment = Experiment(self.settings)

    def run(self):
        self.experiment.run()
