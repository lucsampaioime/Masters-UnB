from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from random import choice
import os

# Inicializa listas para armazenar os valores das métricas
accuracy_values = []
precision_values = []
recall_values = []
f1_values = []
mse_values = []

data_folder = Path(
    "C:/Users/lucsa/Dropbox/Data Science/Mestrado UNB/Dissertação/Experimentos/Testes/NMF/Datasets/")

file_path = data_folder / "CSTR.csv"

# Carrega o dataset
with open(file_path, 'r') as f:
    lines = f.readlines()[1:]  # pula a primeira linha

# Divide cada linha em texto e classe
texts, classes = zip(*[line.rsplit(',', 1) for line in lines])

# Converte as classes para números
le = LabelEncoder()
classes = le.fit_transform(classes)

# Escolhe uma classe aleatória para ser a positiva, e transforma o restante em negativa
positive_class = classes[0]
# positive_class = choice(np.unique(classes))
classes = np.where(classes == positive_class, 1, 0)

# Cria a matrix documento-palavra
vectorizer = TfidfVectorizer()
# vectorizer = CountVectorizer()
V = vectorizer.fit_transform(texts).toarray()


def nmf_with_update(V, n_topics, n_labeled, max_iter, tol):

    # Rotula uma quantidade n_labeled de documentos da classe positiva
    positive_indexes = np.where(classes == 1)[0]
    # np.random.shuffle(positive_indexes)
    labeled = positive_indexes[:n_labeled]
    unlabeled = positive_indexes[n_labeled:]

    # Inicializa as matrizes W e H
    W = np.random.rand(V.shape[0], n_topics)
    H = np.random.rand(n_topics, V.shape[1])

    # Determina qual tópico é o mais representativo para a classe positiva [usar uma das opções abaixo]
    # positive_class_index = np.argmax(np.sum(W[classes == 1], axis=0))
    positive_class_index = 0

    # Cria máscaras booleanas para os documentos rotulados e não rotulados
    labeled_mask = np.zeros(len(classes), dtype=bool)
    labeled_mask[labeled] = True
    unlabeled_mask = ~labeled_mask

    for n in range(max_iter):
        # Atualiza as matrizes W e H - Euclidian
        # W *= (V @ H.T) / (W @ (H @ H.T) + np.finfo(float).eps)
        # H *= (W.T @ V) / ((W.T @ W) @ H + np.finfo(float).eps)

        # Atualiza as matrizes W e H - KL
        W *= (V @ H.T) / (W @ H @ H.T + 1e-10)
        H *= (W.T @ V) / (W.T @ W @ H + 1e-10)

        # Aplique a função 'Supress'
        W[labeled, 0] = np.max(W)
        W[labeled, 1:] = 0.001

        # Calcula a norma euclidiana entre V e WH
        # error = np.linalg.norm(V - W @ H)

        # Calcula a divergencia KL entre V e WH
        eps = 1e-10  # small constant to avoid division by zero
        V = np.maximum(V, eps)
        W_H = np.maximum(W @ H, eps)
        error = np.sum(V * np.log(V / W_H) - V + W_H)

        # Se o erro for menor que a tolerância, para o processo
        if error < tol:
            break

        # Calcula as métricas de classificação
        preds = np.argmax(W, axis=1) == positive_class_index
        preds = preds.astype(int)

        accuracy = accuracy_score(
            classes[unlabeled_mask], preds[unlabeled_mask])
        precision = precision_score(
            classes[unlabeled_mask], preds[unlabeled_mask], average='micro', zero_division=0)
        recall = recall_score(
            classes[unlabeled_mask], preds[unlabeled_mask], average='micro', zero_division=0)
        f1 = f1_score(classes[unlabeled_mask],
                      preds[unlabeled_mask], average='micro', zero_division=0)
        cm = confusion_matrix(classes[unlabeled_mask], preds[unlabeled_mask])

        # Calcula o MSE
        mse = mean_squared_error(V, W @ H)
        log_mse = np.log(mse + 1e-10)
        mse_values.append(log_mse)

    global global_n_topics
    global_n_topics = n_topics

    global global_n_labeled
    global_n_labeled = n_labeled

    # Armazena os valores das métricas
    accuracy_values.append(accuracy)
    precision_values.append(precision)
    recall_values.append(recall)
    f1_values.append(f1)

    return W, H


for x in range(10):

    # nmf_with_update(matrix V, n_topics, n_labeled, max_iter, tol)
    nmf_with_update(V, 50, 10, 30, 1e-4)

# Calcula a média e o desvio-padrão das métricas
accuracy_mean, accuracy_std = np.mean(
    accuracy_values), np.std(accuracy_values)
precision_mean, precision_std = np.mean(
    precision_values), np.std(precision_values)
recall_mean, recall_std = np.mean(recall_values), np.std(recall_values)
f1_mean, f1_std = np.mean(f1_values), np.std(f1_values)

mse_mean, mse_std = np.mean(mse_values), np.std(mse_values)

print(f"Arquivo: {os.path.basename(file_path)}")

print(f"Qtde tópicos: {global_n_topics}")

print(f"Qtde rotulados: {global_n_labeled}")


print(f"Média e desvio padrão da acurácia: {accuracy_mean}, {accuracy_std}")
print(
    f"Média e desvio padrão da precisão: {precision_mean}, {precision_std}")
print(f"Média e desvio padrão do recall: {recall_mean}, {recall_std}")
print(f"Média e desvio padrão do F1 score: {f1_mean}, {f1_std}")
print(f"Média e desvio padrão do log(MSE): {mse_mean}, {mse_std}")
