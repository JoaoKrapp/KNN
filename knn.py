import numpy as np
import pandas as pd

import scipy.io
from typing import List, Union, Tuple, Optional

def load_data(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and prepare the dataset from a MATLAB file.
    """
    mat_data = scipy.io.loadmat(file_path)
    
    train_features = pd.DataFrame(mat_data['grupoTrain'])
    train_labels = pd.DataFrame(mat_data['trainRots'])
    
    test_features = pd.DataFrame(mat_data['grupoTest'])
    test_labels = pd.DataFrame(mat_data['testRots'])
    
    return train_features, train_labels, test_features, test_labels


def euclidean_distance(row_train: np.ndarray, row_test: np.ndarray) -> float:
    """
    Calculate the Euclidean distance between two data points.
    """
    if len(row_train) != len(row_test):
        raise ValueError("Error: Input rows must have the same length.")

    return np.sqrt(np.sum((row_test - row_train)**2))

def get_accuracy(predicted_labels: List[np.uint], true_labels: pd.DataFrame) -> float:
        """ Calculate the classification accuracy by comparing predicted labels with true labels.

        Args:
            predicted_labels (List[np.uint]): List of predicted class labels.
            true_labels (pd.DataFrame): DataFrame containing the true class labels.

        Returns:
            float: The accuracy score (proportion of correct predictions) between 0 and 1.
        """
        
        # Convert true_labels from DataFrame to a flat array for comparison
        true_labels_array = np.array(true_labels).flatten()
        
        # Convert predicted_labels to numpy array if it's not already
        predicted_labels_array = np.array(predicted_labels)
        
        # Check if the lengths match
        if len(predicted_labels_array) != len(true_labels_array):
            raise ValueError("The length of predicted_labels and true_labels must be the same.")
        
        # Calculate accuracy: number of correct predictions divided by total predictions
        correct_predictions = np.sum(predicted_labels_array == true_labels_array)
        total_predictions = len(true_labels_array)
        
        accuracy = correct_predictions / total_predictions
        return accuracy

class KNN:
    
    def __init__(self, 
                train_features : pd.DataFrame, 
                train_labels : pd.DataFrame, 
                test_features : pd.DataFrame, 
                test_labels : pd.DataFrame,
                columns_to_remove : Optional[List[str]] = [],
                columns_only : Optional[List[str]] = []
                ):
        
        self.train_features = train_features
        self.train_labels = train_labels
        self.test_features = test_features
        self.test_labels = test_labels
        
        if columns_to_remove:
            self._drop_columns(columns_to_remove)
            
        if columns_only:
            self._only_columns(columns_only)
        
        self.dist = self._generate_distance_matrix()
        
    def get_predictions(self, k : int) -> np.array:
        predictions = []
        
        for row_dist in self.dist:
            # Indice dos train_labels mais pertos
            neighbors_indices = np.argsort(row_dist)[:k]
            
            # Valor dos vizinhos
            neighbors = self.train_labels.iloc[neighbors_indices]
            
            # Valor mais comum dos vizinhos
            pred = neighbors[0].value_counts().index[0]
            predictions.append(pred)
            
        return np.array(predictions)
    
    def run_range_k(self, stop : int):
        data = {
            'k' : [],
            'accuracy' : []
        }
        
        for i in range(1, stop):
            pred = self.get_predictions(i)
            acc = get_accuracy(predicted_labels=pred, true_labels=self.test_labels)

            data['k'].append(i)
            data['accuracy'].append(acc)

        return data
    
    def _only_columns(self, columns):
        self.train_features = self.train_features[columns]
        self.test_features = self.test_features[columns]
    
    def _drop_columns(self, columns : List[str]):
        
        for col in columns:
            self.train_features.drop(col, axis=1, inplace=True)
            self.test_features.drop(col, axis=1, inplace=True)
        
    def _generate_distance_matrix(self) -> np.array:
        
        dist = []
        
        for index_test, test_row in self.test_features.iterrows():
            
            row_distances = []
            
            for index_training, training_row in self.train_features.iterrows():
                element_distance = euclidean_distance(training_row, test_row)
                row_distances.append(element_distance)
                
            dist.append(np.array(row_distances))
            
        return np.array(dist)
