import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from random import choice
import os

# Inicializa listas para armazenar os valores das métricas
accuracy_values = []
precision_values = []
recall_values = []
f1_values = []


def nmf_with_update(file_path, n_topics, n_labeled, max_iter, tol):
    # Carrega o dataset
    with open(file_path, 'r') as f:
        lines = f.readlines()[1:]  # pula a primeira linha

    # Divide cada linha em texto e classe
    texts, classes = zip(*[line.rsplit(',', 1) for line in lines])

    # Converte as classes para números
    le = LabelEncoder()
    classes = le.fit_transform(classes)

    # Escolha uma classe aleatória para ser a positiva, e transforme o restante em negativa
    # positive_class = classes[0]
    positive_class = choice(np.unique(classes))
    classes = np.where(classes == positive_class, 1, 0)

    # Rotula uma quantidade n_labeled de documentos da classe positiva
    positive_indexes = np.where(classes == 1)[0]
    np.random.shuffle(positive_indexes)
    labeled = positive_indexes[:n_labeled]
    unlabeled = positive_indexes[n_labeled:]

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
        W[labeled, 0] = np.max(W)
        W[labeled, 1:] = 0.001

        # Calcula a norma euclidiana entre V e WH
        error = np.linalg.norm(V - W @ H)

        # Se o erro for menor que a tolerância, para o processo
        if error < tol:
            break

        # print(f"Iteração {n+1}: erro = {error}")
        # Calcula as métricas de classificação
        preds = np.argmax(W, axis=1)
        accuracy = accuracy_score(classes, preds)
        precision = precision_score(
            classes, preds, average='weighted', zero_division=0)
        recall = recall_score(
            classes, preds, average='weighted', zero_division=0)
        f1 = f1_score(classes, preds, average='weighted', zero_division=0)
        cm = confusion_matrix(classes, preds)

    print(f"Arquivo: {os.path.basename(file_path)}")

    print(f"Qtde rotulados: {n_labeled}")

    print(f"Acurácia: {accuracy}")
    print(f"Precisão: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 score: {f1}")
    print(f"Matriz de Confusão na Iteração {n+1}: \n {cm}")

    # Armazena os valores das métricas
    accuracy_values.append(accuracy)
    precision_values.append(precision)
    recall_values.append(recall)
    f1_values.append(f1)

    return W, H


for x in range(10):

    print(f"Resultado nº {x}")

    # nmf_with_update(file_path, n_topics, n_labeled, max_iter, tol)
    nmf_with_update(
        'C:/Users/lucsa/Dropbox/Data Science/Mestrado UNB/Dissertação/Experimentos/Testes/NMF/Datasets/Reuters-21578.csv', 10, 10, 30, 1e-4)

    # Calcula a média e o desvio-padrão das métricas
    accuracy_mean, accuracy_std = np.mean(
        accuracy_values), np.std(accuracy_values)
    precision_mean, precision_std = np.mean(
        precision_values), np.std(precision_values)
    recall_mean, recall_std = np.mean(recall_values), np.std(recall_values)
    f1_mean, f1_std = np.mean(f1_values), np.std(f1_values)


print(f"Média e desvio padrão da acurácia: {accuracy_mean}, {accuracy_std}")
print(
    f"Média e desvio padrão da precisão: {precision_mean}, {precision_std}")
print(f"Média e desvio padrão do recall: {recall_mean}, {recall_std}")
print(f"Média e desvio padrão do F1 score: {f1_mean}, {f1_std}")
