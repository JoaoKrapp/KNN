from knn import KNN, load_data

train_features, train_labels, test_features, test_labels = load_data('./dados/grupoDados2.mat')
    
knn = KNN(train_features, train_labels, test_features, test_labels, columns_only=[12])

a = knn.run_range_k(15)
print(a)