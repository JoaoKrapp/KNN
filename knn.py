import scipy.io
import pandas as pd
import numpy as np
from typing import List, Union, Tuple

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

def indices_of_n_smallest(array: np.ndarray, n: int) -> np.ndarray:
    """
    Find the indices of the n smallest values in an array.
    """
    # np.argsort returns the indices that would sort the array
    indices = np.argsort(array)
    
    # Returns the first n indices (which correspond to the n smallest values)
    return indices[:n]

class KNN:
    """
    K-Nearest Neighbors (KNN) classifier implementation.
    """
    
    def __init__(self, training_features: pd.DataFrame, training_labels: pd.DataFrame):
        """
        Initialize the KNN classifier with training data and labels.
        """
        self.training_features = training_features
        self.training_labels = training_labels
        self.distance_matrix = []
        
    def _calculate_distances(self, query_point: pd.Series) -> List[float]:
        """ Calculate distances from the query point to all training data points.

        Args:
            query_point (pd.Series): The data point to calculate distances.

        Returns:
            List[float]: List of distances from the query point to each training data point.
        """
        distances = []
  
        for index, training_row in self.training_features.iterrows():
            distance = euclidean_distance(training_row, query_point)
            distances.append(distance)
   
        return distances
    
    def _get_nearest_neighbors(self, distances: List[float], k: int) -> pd.DataFrame:
        """ Find the k nearest neighbors based on the distances.

        Args:
            distances (List[float]): List of distances from the query point to each training data point.
            k (int): Number of nearest neighbors to consider.

        Returns:
            pd.DataFrame: Labels of the k nearest neighbors.
        """
        nearest_indices = indices_of_n_smallest(distances, k)
        nearest_labels = self.training_labels.iloc[nearest_indices]
        return nearest_labels
    
    def _predict_label(self, neighbor_labels: pd.DataFrame) -> int:
        """ Predict the class label based on the majority vote of nearest neighbors.

        Args:
            neighbor_labels (pd.DataFrame): Labels of the nearest neighbors.

        Returns:
            int: The predicted class label.
        """
        values, counts = np.unique(neighbor_labels, return_counts=True)
        most_common_index = np.argmax(counts)
        return values[most_common_index]
    
    def predict(self, query_point: pd.Series, k: int) -> int:
        """ Predict the class of a query point using k-nearest neighbors.

        Args:
            query_point (pd.Series): The data point to classify.
            k (int): Number of nearest neighbors to consider.

        Returns:
            int: The predicted class label.
        """
        
        if k < 1:
            raise ValueError("Error: Value K can't have value below 1")
        elif k > len(self.training_features):
            raise ValueError("Error: Not enough neightbors(K value too big)")
        
        distances = self._calculate_distances(query_point)
        nearest_neighbors = self._get_nearest_neighbors(distances, k)
        predicted_label = self._predict_label(nearest_neighbors)
        return predicted_label
    
    def predict_all(self, data : pd.DataFrame, k: int):
        """ Predict labels for all data points in a test DataFrame.

        Args:
            data (pd.DataFrame): DataFrame containing the test data points to classify.
            k (int): Number of nearest neighbors to consider.

        Returns:
            List[int]: Array of predicted labels for each test data point.
        """
        predicted_labels = []
        
        # For each row predict
        for index, row in data.iterrows():
            predicted_labels.append(self.predict(row, k))
            
        return predicted_labels
    
    def get_accuracy(self, predicted_labels: List[np.uint], true_labels: pd.DataFrame) -> float:
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
    
    def generate_distance_matrix(self, data : pd.DataFrame) -> List[List[np.float64]]:
        """ Calculate distances from the all query points in data to all training data points, and stores it

        Args:
            data (pd.DataFrame): DataFrame containing the test data points to classify.

        Returns:
            List[List[np.float64]]: Two dimentional array with distances. Example: Distance between the variable "data[0]" and "self.training_features[1]" is "self.distance_matix[0][1]"
        """
        for index, row in data.iterrows():
            distance = self._calculate_distances(row)
            self.distance_matrix.append(distance)
            
        return self.distance_matrix
    
    def get_prediction_from_distance_matrix(self, k : int, test_feature_index : int) -> int:
        """ Get the predicted label with values in "self.distance_matrix"

        Args:
            k (int): Number of nearest neighbors to consider.
            test_feature_index (int): Index of test_feature used to calculate distances of "self.distance_matrix"

        Returns:
            int: Predicted label
        """
        distances = self.distance_matrix[test_feature_index]
        nearest_neighbors = self._get_nearest_neighbors(distances, k)
        return self._predict_label(nearest_neighbors)
    
    def get_all_predictions_from_distance_matrix(self, k : int) -> List[int]:
        """ Get all the predictions of labels from "self.distance_matrix"

        Args:
            k (int): Number of nearest neighbors to consider.

        Returns:
            List[int]: List of predicted labels
        """
        labels = []
        for index in range(len(self.distance_matrix)):
            prediction = self.get_prediction_from_distance_matrix(k, index)
            labels.append(prediction)
            
        return labels
        