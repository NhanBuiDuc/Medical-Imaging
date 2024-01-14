from base import BaseVisualizer
# Implement your preferred way of visualizing the class distribution
# For example, using matplotlib
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
# Example subclass


class ClassDistributionVisualizer(BaseVisualizer):
    def visualize(self):
        class_distribution = self.data_loader.dataset.class_distribution
        self.plot_class_distribution(class_distribution)
        print(class_distribution)

    def plot_class_distribution(self, class_distribution):

        classes = list(class_distribution.keys())
        counts = list(class_distribution.values())

        plt.bar(classes, counts)
        plt.xlabel('Class')
        plt.ylabel('Number of Samples')
        plt.title('Class Distribution')
        plt.show()
