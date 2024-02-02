import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder


def nmf_with_update(file_path, n_topics, n_labeled, max_iter, tol):
    # Carrega o dataset
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Divide cada linha em texto e classe
    texts, classes = zip(*[line.rsplit(',', 1) for line in lines])

    # Converte as classes para números
    le = LabelEncoder()
    classes = le.fit_transform(classes)

    # Cria a matrix documento-palavra
    vectorizer = CountVectorizer()
    V = vectorizer.fit_transform(texts).toarray()

    # Inicializa as matrizes W e H
    W = np.random.rand(V.shape[0], n_topics)
    H = np.random.rand(n_topics, V.shape[1])

    for n in range(max_iter):
        # Atualiza as matrizes W e H
        W *= (V @ H.T) / (W @ (H @ H.T) + np.finfo(float).eps)
        H *= (W.T @ V) / ((W.T @ W) @ H + np.finfo(float).eps)

        # Aplique a função 'Supress'
        W[:n_labeled, 0] = np.max(W)
        W[:n_labeled, 1:] = 0.001

        # Calcula a norma euclidiana entre V e WH
        error = np.linalg.norm(V - W @ H)

        # Se o erro for menor que a tolerância, para o processo
        if error < tol:
            break
        print(f"Iteração {n+1}:")
        print(f"Erro = {error}")

        # Calcula as métricas de classificação
        preds = np.argmax(W, axis=1)
        print(f"Acurácia: {accuracy_score(classes, preds)}")
        print(
            f"Precisão: {precision_score(classes, preds, average='weighted')}")
        print(
            f"Recall: {recall_score(classes, preds, average='weighted', zero_division=0)}")
        print(f"F1 score: {f1_score(classes, preds, average='weighted')}")

    return W, H


nmf_with_update(
    'C:/Users/lucsa/Dropbox/Data Science/Mestrado UNB/Dissertação/Experimentos/Testes/NMF/Datasets/CSTR.csv', 10, 30, 30, 1e-4)
