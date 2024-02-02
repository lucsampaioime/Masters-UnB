import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def nmf_with_update(file_path, n_topics, n_labeled, max_iter=2, tol=1e-4):
    # Carrega o dataset
    data = pd.read_csv(file_path, names=['document'], header=None)
    data[['Text', 'Class']] = data['document'].str.rsplit(
        ',', n=1, expand=True)

    # Cria a matriz documento-palavra V
    vectorizer = CountVectorizer()
    V = vectorizer.fit_transform(data['Text']).toarray()
    V = V.astype(float)
    print(V)

    # Classifica os documentos
    data['is_positive'] = data['Class'] == 'positive'
    data['is_labeled'] = False
    data.loc[data['is_positive'], 'is_labeled'] = range(
        sum(data['is_positive']))
    data['is_labeled'] = data['is_labeled'] < n_labeled

    # Inicializa as matrizes W e H
    np.random.seed(123)
    W = np.random.rand(V.shape[0], n_topics)
    H = np.random.rand(n_topics, V.shape[1])

    for n in range(max_iter):
        # Atualização multiplicativa de W e H
        WH = np.dot(W, H) + 1e-10
        ratio = V / WH
        H = H * (np.dot(W.T, ratio) / np.sum(W, axis=0).reshape((n_topics, -1)))
        WH = np.dot(W, H) + 1e-10
        ratio = V / WH
        W = W * (np.dot(ratio, H.T) / np.sum(H, axis=1))

        # Força o primeiro valor da linha d(i) na matriz W a ser o valor máximo para documentos rotulados
        max_val = np.max(W)
        for i in data[data['is_labeled']].index:
            W[i] = 0.001
            W[i][0] = max_val

        # Verifica a convergência
        WH = np.dot(W, H)
        e = np.sqrt(np.sum((V - WH) ** 2))
        print(f"Iteração {n+1}: erro = {e}")

        # Cálculo de acurácia, precisão, recall e f1 score
        predicted_classes = np.argmax(W, axis=1)
        true_classes = data['Class'].apply(
            lambda x: 1 if x == 'positive' else 0).values

        accuracy = accuracy_score(true_classes, predicted_classes)
        precision = precision_score(
            true_classes, predicted_classes, zero_division=1)
        recall = recall_score(true_classes, predicted_classes, zero_division=1)
        f1 = f1_score(true_classes, predicted_classes)

        print(
            f"Iteração {n+1}: acurácia = {accuracy}, precisão = {precision}, recall = {recall}, f1 score = {f1}")

        if e < tol:
            break

    return W, H


# Exemplo de uso
W, H = nmf_with_update(
    'C:/Users/lucsa/Dropbox/Data Science/Mestrado UNB/Dissertação/Experimentos/Testes/NMF/Datasets/Reviews.csv', n_topics=10, n_labeled=5)
