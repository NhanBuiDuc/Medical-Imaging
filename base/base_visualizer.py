
import numpy as np
from abc import ABC, abstractmethod


class BaseVisualizer(ABC):
    def __init__(self, split_visualization):
        self.data_loader = None
        self.split_visualization = split_visualization

    @abstractmethod
    def visualize(self):
        """
        Abstract method to visualize information about the dataset
        """
        pass

    def __str__(self):
        """
        String representation of the visualizer, can be overridden by subclasses
        """
        return super().__str__() + '\nAdditional information about the visualizer'


# Example usage:
# visualizer = ClassDistributionVisualizer(your_data_loader)
# visualizer.visualize()
