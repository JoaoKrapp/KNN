from knn import KNN, load_data

train_features, train_labels, test_features, test_labels = load_data('./dados/grupoDados1.mat')


# Q1.1. Qual é a acurácia máxima que você consegue da classificação?
# R: 0.98

def questao1():
    knn = KNN(train_features, train_labels)

    for k in range(1, 15):
        predictions = knn.predict_all(test_features, k)
        accuracy = knn.get_accuracy(predictions, test_labels)
        print(f"A acuracia com k={k} é de {accuracy}")

# Q1.2. É necessário ter todas as características (atributos) para obter a acurácia máxima para esta classificação?
# R:
# A acuracia removando a coluna 0 for de: 0.9514285714285713
# A acuracia removando a coluna 1 for de: 0.9571428571428573
# A acuracia removando a coluna 2 for de: 0.9285714285714286
# TODO: remover todas as colunas, range ta errado

def questao2():
    
    for i in train_features.columns:
        print(f"Removendo a coluna {i}")
        
        train_features_column_removed = train_features.drop(i, axis=1)
        test_features_column_removed = test_features.drop(i, axis=1)
        
        knn = KNN(train_features_column_removed, train_labels)
        acuracias = []
        
        for k in range(1, 15):
            predictions = knn.predict_all(test_features_column_removed, k)
            accuracy = knn.get_accuracy(predictions, test_labels)
            acuracias.append(accuracy)
            
        print(f"A acuracia removando a coluna {i} for de:")
        print(sum(acuracias) / len(acuracias))
            
questao2()