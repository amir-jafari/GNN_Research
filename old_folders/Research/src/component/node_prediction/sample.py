# Code Reference: https://github.com/amir-jafari/Capstone/tree/main/Sample_Capstone

# -*- coding: utf-8 -*-
"""
Author: Shikha Kumari
Date: 2024-10-21
Version: 1.0
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from utils import remove_columns
# import networkx as nx

class IrisDataExplorer:
    def __init__(self, file_path):
        """
        :rtype: object
        :param file_path: str - Path to the CSV file containing the Iris dataset
        """
        self.data = pd.read_csv(file_path)
        self.graph = None

    def display_data(self):
        """
        :rtype: None
        :return: Prints the first few rows and unique species in the dataset.
        """
        print(self.data.head())
        print("\nUnique Species:")
        print(self.data['Species'].unique())

    def describe_data(self): # TODO: move this to utils
        """
        :rtype: None
        :return: Prints summary statistics like mean, standard deviation, etc., for numerical columns.
        """
        print("\nData Description:")
        print(self.data.describe())

    def visualize_data(self, cols_to_drop): # TODO: move this to utils
        """
        :rtype: None
        :return: Displays a seaborn pairplot to visualize feature relationships with respect to species.
        """

        data = remove_columns(self.data, cols_to_drop)
        sns.pairplot(data, hue='Species')
        plt.show()

    def encode_labels(self): # TODO: move this function to utils
        """
        :rtype: None
        :return: Adds a new column 'Species_Encoded' to the dataset with encoded labels.
        """
        le = LabelEncoder()
        self.data['Species_Encoded'] = le.fit_transform(self.data['Species'])
        print("\nEncoded Species Labels:")
        print(self.data[['Species', 'Species_Encoded']].drop_duplicates())

    def get_node_features(self):
        """
        Extract node features from the dataset, excluding 'Id' and 'Species' columns.
        :rtype: numpy.ndarray
        :return: A NumPy array containing node features (all columns except 'Id' and 'Species').
        """
        node_features = self.data.iloc[:, 1:-1].values.astype(float)
        print("\nNode Features):")
        print(node_features)
        return node_features
    
    @staticmethod
    def main(file_path):
        """
        Main function to run the exploration workflow for the Iris dataset.

        :rtype: None
        :param file_path: str - Path to the CSV file containing the Iris dataset.
        """
        explorer = IrisDataExplorer(file_path)
        
        # Step 1: Display the dataset
        print("Step 1: Displaying the dataset:")
        explorer.display_data()

        # Step 2: Describe the dataset
        print("\nStep 2: Describing the dataset:")
        explorer.describe_data()

        # Step 3: Visualize the dataset
        print("\nStep 3: Visualizing the dataset:")
        columns_to_drop = ['Id', 'Species']
        explorer.visualize_data(columns_to_drop)

        # Step 4: Encode species labels
        print("\nStep 4: Encoding species labels:")
        explorer.encode_labels()

        # Step 5: Extract node features
        print("\nStep 5: Extracting node features:")
        explorer.get_node_features()

# Example usage
if __name__ == "__main__":
    # TODO: Add argparser
    file_path = 'gnn/src/component/node_prediction/data/Iris.csv'
    IrisDataExplorer.main(file_path)
