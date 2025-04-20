from knn import KNN, load_data
import numpy as np

def main():
    train_features, train_labels, test_features, test_labels = load_data('./dados/grupoDados1.mat')

    knn = KNN(train_features, train_labels)
    knn.generate_distance_matrix(test_features)
    
    # Q1.1. Qual é a acurácia máxima que você consegue da classificação?
    for k in range(1, 10):
        predicted_labels = knn.get_all_predictions_from_distance_matrix(k)
        accuracy = knn.get_accuracy(predicted_labels, test_labels)
        print(f"Acuracia com k={k} é de {accuracy}")
    # print(predicted_labels)
    
if __name__ == '__main__':
    main()