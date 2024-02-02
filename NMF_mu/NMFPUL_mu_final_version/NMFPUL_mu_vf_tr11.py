from pathlib import Path
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

data_folder = Path(
    "C:/Users/lucsa/Dropbox/Data Science/Mestrado UNB/Dissertação/Experimentos/Testes/NMF/Datasets/")


def nmf_with_update(file_path, n_topics, n_labeled, max_iter, tol):
    # Carrega o dataset
    with open(file_path, 'r') as f:
        lines = f.readlines()[1:]  # pula a primeira linha

    # Divide cada linha em texto e classe
    texts, classes = zip(*[line.rsplit(',', 1) for line in lines])

    # Converte as classes para números
    le = LabelEncoder()
    classes = le.fit_transform(classes)

    # Escolhe uma classe aleatória para ser a positiva, e transforma o restante em negativa
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

        # print(f"Iteração {n+1}: erro = {error}")

        # Calcula as métricas de classificação
        preds = np.argmax(W, axis=1) == positive_class_index
        preds = preds.astype(int)

        # accuracy = accuracy_score(classes, preds)
        # precision = precision_score(
        #     classes, preds, average='macro', zero_division=0)
        # recall = recall_score(
        #     classes, preds, average='macro', zero_division=0)
        # f1 = f1_score(classes, preds, average='macro', zero_division=0)
        # cm = confusion_matrix(classes, preds)

        accuracy = accuracy_score(
            classes[unlabeled_mask], preds[unlabeled_mask])
        precision = precision_score(
            classes[unlabeled_mask], preds[unlabeled_mask], average='weighted', zero_division=0)
        recall = recall_score(
            classes[unlabeled_mask], preds[unlabeled_mask], average='weighted', zero_division=0)
        f1 = f1_score(classes[unlabeled_mask], preds[unlabeled_mask],
                      average='weighted', zero_division=0)
        cm = confusion_matrix(classes[unlabeled_mask], preds[unlabeled_mask])

    global global_file_path
    global_file_path = os.path.basename(file_path)

    global global_n_topics
    global_n_topics = n_topics

    global global_n_labeled
    global_n_labeled = n_labeled

    # print(f"Arquivo: {os.path.basename(file_path)}")
    # print(f"Qtde rotulados: {n_labeled}")
    # print(f"Acurácia: {accuracy}")
    # print(f"Precisão: {precision}")
    # print(f"Recall: {recall}")
    # print(f"F1 score: {f1}")
    # print(f"Matriz de Confusão na Iteração {n+1}: \n {cm}")

    # Armazena os valores das métricas
    accuracy_values.append(accuracy)
    precision_values.append(precision)
    recall_values.append(recall)
    f1_values.append(f1)

    return W, H


for x in range(10):

    # print(f"Resultado nº {x+1}")

    # nmf_with_update(file_path, n_topics, n_labeled, max_iter, tol)
    nmf_with_update(data_folder / "tr11.mat.csv", 5, 1, 300, 1e-4)

# Calcula a média e o desvio-padrão das métricas
accuracy_mean, accuracy_std = np.mean(
    accuracy_values), np.std(accuracy_values)
precision_mean, precision_std = np.mean(
    precision_values), np.std(precision_values)
recall_mean, recall_std = np.mean(recall_values), np.std(recall_values)
f1_mean, f1_std = np.mean(f1_values), np.std(f1_values)

print(f"Arquivo: {global_file_path}")

print(f"Qtde tópicos: {global_n_topics}")

print(f"Qtde rotulados: {global_n_labeled}")


print(f"Média e desvio padrão da acurácia: {accuracy_mean}, {accuracy_std}")
print(
    f"Média e desvio padrão da precisão: {precision_mean}, {precision_std}")
print(f"Média e desvio padrão do recall: {recall_mean}, {recall_std}")
print(f"Média e desvio padrão do F1 score: {f1_mean}, {f1_std}")

for x in range(10):

    # print(f"Resultado nº {x+1}")

    # nmf_with_update(file_path, n_topics, n_labeled, max_iter, tol)
    nmf_with_update(data_folder / "tr11.mat.csv", 10, 1, 300, 1e-4)

# Calcula a média e o desvio-padrão das métricas
accuracy_mean, accuracy_std = np.mean(
    accuracy_values), np.std(accuracy_values)
precision_mean, precision_std = np.mean(
    precision_values), np.std(precision_values)
recall_mean, recall_std = np.mean(recall_values), np.std(recall_values)
f1_mean, f1_std = np.mean(f1_values), np.std(f1_values)

print(f"Arquivo: {global_file_path}")

print(f"Qtde tópicos: {global_n_topics}")

print(f"Qtde rotulados: {global_n_labeled}")

print(f"Média e desvio padrão da acurácia: {accuracy_mean}, {accuracy_std}")
print(
    f"Média e desvio padrão da precisão: {precision_mean}, {precision_std}")
print(f"Média e desvio padrão do recall: {recall_mean}, {recall_std}")
print(f"Média e desvio padrão do F1 score: {f1_mean}, {f1_std}")

for x in range(10):

    # print(f"Resultado nº {x+1}")

    # nmf_with_update(file_path, n_topics, n_labeled, max_iter, tol)
    nmf_with_update(data_folder / "tr11.mat.csv", 20, 1, 300, 1e-4)

# Calcula a média e o desvio-padrão das métricas
accuracy_mean, accuracy_std = np.mean(
    accuracy_values), np.std(accuracy_values)
precision_mean, precision_std = np.mean(
    precision_values), np.std(precision_values)
recall_mean, recall_std = np.mean(recall_values), np.std(recall_values)
f1_mean, f1_std = np.mean(f1_values), np.std(f1_values)

print(f"Arquivo: {global_file_path}")

print(f"Qtde tópicos: {global_n_topics}")

print(f"Qtde rotulados: {global_n_labeled}")

print(f"Média e desvio padrão da acurácia: {accuracy_mean}, {accuracy_std}")
print(
    f"Média e desvio padrão da precisão: {precision_mean}, {precision_std}")
print(f"Média e desvio padrão do recall: {recall_mean}, {recall_std}")
print(f"Média e desvio padrão do F1 score: {f1_mean}, {f1_std}")

for x in range(10):

    # print(f"Resultado nº {x+1}")

    # nmf_with_update(file_path, n_topics, n_labeled, max_iter, tol)
    nmf_with_update(data_folder / "tr11.mat.csv", 50, 1, 300, 1e-4)

# Calcula a média e o desvio-padrão das métricas
accuracy_mean, accuracy_std = np.mean(
    accuracy_values), np.std(accuracy_values)
precision_mean, precision_std = np.mean(
    precision_values), np.std(precision_values)
recall_mean, recall_std = np.mean(recall_values), np.std(recall_values)
f1_mean, f1_std = np.mean(f1_values), np.std(f1_values)

print(f"Arquivo: {global_file_path}")

print(f"Qtde tópicos: {global_n_topics}")

print(f"Qtde rotulados: {global_n_labeled}")

print(f"Média e desvio padrão da acurácia: {accuracy_mean}, {accuracy_std}")
print(
    f"Média e desvio padrão da precisão: {precision_mean}, {precision_std}")
print(f"Média e desvio padrão do recall: {recall_mean}, {recall_std}")
print(f"Média e desvio padrão do F1 score: {f1_mean}, {f1_std}")


###############################################################################


for x in range(10):

    # print(f"Resultado nº {x+1}")

    # nmf_with_update(file_path, n_topics, n_labeled, max_iter, tol)
    nmf_with_update(data_folder / "tr11.mat.csv", 5, 5, 300, 1e-4)

# Calcula a média e o desvio-padrão das métricas
accuracy_mean, accuracy_std = np.mean(
    accuracy_values), np.std(accuracy_values)
precision_mean, precision_std = np.mean(
    precision_values), np.std(precision_values)
recall_mean, recall_std = np.mean(recall_values), np.std(recall_values)
f1_mean, f1_std = np.mean(f1_values), np.std(f1_values)

print(f"Arquivo: {global_file_path}")

print(f"Qtde tópicos: {global_n_topics}")

print(f"Qtde rotulados: {global_n_labeled}")


print(f"Média e desvio padrão da acurácia: {accuracy_mean}, {accuracy_std}")
print(
    f"Média e desvio padrão da precisão: {precision_mean}, {precision_std}")
print(f"Média e desvio padrão do recall: {recall_mean}, {recall_std}")
print(f"Média e desvio padrão do F1 score: {f1_mean}, {f1_std}")

for x in range(10):

    # print(f"Resultado nº {x+1}")

    # nmf_with_update(file_path, n_topics, n_labeled, max_iter, tol)
    nmf_with_update(data_folder / "tr11.mat.csv", 10, 5, 300, 1e-4)

# Calcula a média e o desvio-padrão das métricas
accuracy_mean, accuracy_std = np.mean(
    accuracy_values), np.std(accuracy_values)
precision_mean, precision_std = np.mean(
    precision_values), np.std(precision_values)
recall_mean, recall_std = np.mean(recall_values), np.std(recall_values)
f1_mean, f1_std = np.mean(f1_values), np.std(f1_values)

print(f"Arquivo: {global_file_path}")

print(f"Qtde tópicos: {global_n_topics}")

print(f"Qtde rotulados: {global_n_labeled}")

print(f"Média e desvio padrão da acurácia: {accuracy_mean}, {accuracy_std}")
print(
    f"Média e desvio padrão da precisão: {precision_mean}, {precision_std}")
print(f"Média e desvio padrão do recall: {recall_mean}, {recall_std}")
print(f"Média e desvio padrão do F1 score: {f1_mean}, {f1_std}")

for x in range(10):

    # print(f"Resultado nº {x+1}")

    # nmf_with_update(file_path, n_topics, n_labeled, max_iter, tol)
    nmf_with_update(data_folder / "tr11.mat.csv", 20, 5, 300, 1e-4)

# Calcula a média e o desvio-padrão das métricas
accuracy_mean, accuracy_std = np.mean(
    accuracy_values), np.std(accuracy_values)
precision_mean, precision_std = np.mean(
    precision_values), np.std(precision_values)
recall_mean, recall_std = np.mean(recall_values), np.std(recall_values)
f1_mean, f1_std = np.mean(f1_values), np.std(f1_values)

print(f"Arquivo: {global_file_path}")

print(f"Qtde tópicos: {global_n_topics}")

print(f"Qtde rotulados: {global_n_labeled}")

print(f"Média e desvio padrão da acurácia: {accuracy_mean}, {accuracy_std}")
print(
    f"Média e desvio padrão da precisão: {precision_mean}, {precision_std}")
print(f"Média e desvio padrão do recall: {recall_mean}, {recall_std}")
print(f"Média e desvio padrão do F1 score: {f1_mean}, {f1_std}")

for x in range(10):

    # print(f"Resultado nº {x+1}")

    # nmf_with_update(file_path, n_topics, n_labeled, max_iter, tol)
    nmf_with_update(data_folder / "tr11.mat.csv", 50, 5, 300, 1e-4)

# Calcula a média e o desvio-padrão das métricas
accuracy_mean, accuracy_std = np.mean(
    accuracy_values), np.std(accuracy_values)
precision_mean, precision_std = np.mean(
    precision_values), np.std(precision_values)
recall_mean, recall_std = np.mean(recall_values), np.std(recall_values)
f1_mean, f1_std = np.mean(f1_values), np.std(f1_values)

print(f"Arquivo: {global_file_path}")

print(f"Qtde tópicos: {global_n_topics}")

print(f"Qtde rotulados: {global_n_labeled}")

print(f"Média e desvio padrão da acurácia: {accuracy_mean}, {accuracy_std}")
print(
    f"Média e desvio padrão da precisão: {precision_mean}, {precision_std}")
print(f"Média e desvio padrão do recall: {recall_mean}, {recall_std}")
print(f"Média e desvio padrão do F1 score: {f1_mean}, {f1_std}")


############################################################################

for x in range(10):

    # print(f"Resultado nº {x+1}")

    # nmf_with_update(file_path, n_topics, n_labeled, max_iter, tol)
    nmf_with_update(data_folder / "tr11.mat.csv", 5, 10, 300, 1e-4)

# Calcula a média e o desvio-padrão das métricas
accuracy_mean, accuracy_std = np.mean(
    accuracy_values), np.std(accuracy_values)
precision_mean, precision_std = np.mean(
    precision_values), np.std(precision_values)
recall_mean, recall_std = np.mean(recall_values), np.std(recall_values)
f1_mean, f1_std = np.mean(f1_values), np.std(f1_values)

print(f"Arquivo: {global_file_path}")

print(f"Qtde tópicos: {global_n_topics}")

print(f"Qtde rotulados: {global_n_labeled}")


print(f"Média e desvio padrão da acurácia: {accuracy_mean}, {accuracy_std}")
print(
    f"Média e desvio padrão da precisão: {precision_mean}, {precision_std}")
print(f"Média e desvio padrão do recall: {recall_mean}, {recall_std}")
print(f"Média e desvio padrão do F1 score: {f1_mean}, {f1_std}")

for x in range(10):

    # print(f"Resultado nº {x+1}")

    # nmf_with_update(file_path, n_topics, n_labeled, max_iter, tol)
    nmf_with_update(data_folder / "tr11.mat.csv", 10, 10, 300, 1e-4)

# Calcula a média e o desvio-padrão das métricas
accuracy_mean, accuracy_std = np.mean(
    accuracy_values), np.std(accuracy_values)
precision_mean, precision_std = np.mean(
    precision_values), np.std(precision_values)
recall_mean, recall_std = np.mean(recall_values), np.std(recall_values)
f1_mean, f1_std = np.mean(f1_values), np.std(f1_values)

print(f"Arquivo: {global_file_path}")

print(f"Qtde tópicos: {global_n_topics}")

print(f"Qtde rotulados: {global_n_labeled}")

print(f"Média e desvio padrão da acurácia: {accuracy_mean}, {accuracy_std}")
print(
    f"Média e desvio padrão da precisão: {precision_mean}, {precision_std}")
print(f"Média e desvio padrão do recall: {recall_mean}, {recall_std}")
print(f"Média e desvio padrão do F1 score: {f1_mean}, {f1_std}")

for x in range(10):

    # print(f"Resultado nº {x+1}")

    # nmf_with_update(file_path, n_topics, n_labeled, max_iter, tol)
    nmf_with_update(data_folder / "tr11.mat.csv", 20, 10, 300, 1e-4)

# Calcula a média e o desvio-padrão das métricas
accuracy_mean, accuracy_std = np.mean(
    accuracy_values), np.std(accuracy_values)
precision_mean, precision_std = np.mean(
    precision_values), np.std(precision_values)
recall_mean, recall_std = np.mean(recall_values), np.std(recall_values)
f1_mean, f1_std = np.mean(f1_values), np.std(f1_values)

print(f"Arquivo: {global_file_path}")

print(f"Qtde tópicos: {global_n_topics}")

print(f"Qtde rotulados: {global_n_labeled}")

print(f"Média e desvio padrão da acurácia: {accuracy_mean}, {accuracy_std}")
print(
    f"Média e desvio padrão da precisão: {precision_mean}, {precision_std}")
print(f"Média e desvio padrão do recall: {recall_mean}, {recall_std}")
print(f"Média e desvio padrão do F1 score: {f1_mean}, {f1_std}")

for x in range(10):

    # print(f"Resultado nº {x+1}")

    # nmf_with_update(file_path, n_topics, n_labeled, max_iter, tol)
    nmf_with_update(data_folder / "tr11.mat.csv", 50, 10, 300, 1e-4)

# Calcula a média e o desvio-padrão das métricas
accuracy_mean, accuracy_std = np.mean(
    accuracy_values), np.std(accuracy_values)
precision_mean, precision_std = np.mean(
    precision_values), np.std(precision_values)
recall_mean, recall_std = np.mean(recall_values), np.std(recall_values)
f1_mean, f1_std = np.mean(f1_values), np.std(f1_values)

print(f"Arquivo: {global_file_path}")

print(f"Qtde tópicos: {global_n_topics}")

print(f"Qtde rotulados: {global_n_labeled}")

print(f"Média e desvio padrão da acurácia: {accuracy_mean}, {accuracy_std}")
print(
    f"Média e desvio padrão da precisão: {precision_mean}, {precision_std}")
print(f"Média e desvio padrão do recall: {recall_mean}, {recall_std}")
print(f"Média e desvio padrão do F1 score: {f1_mean}, {f1_std}")

#############################################################################

for x in range(10):

    # print(f"Resultado nº {x+1}")

    # nmf_with_update(file_path, n_topics, n_labeled, max_iter, tol)
    nmf_with_update(data_folder / "tr11.mat.csv", 5, 20, 300, 1e-4)

# Calcula a média e o desvio-padrão das métricas
accuracy_mean, accuracy_std = np.mean(
    accuracy_values), np.std(accuracy_values)
precision_mean, precision_std = np.mean(
    precision_values), np.std(precision_values)
recall_mean, recall_std = np.mean(recall_values), np.std(recall_values)
f1_mean, f1_std = np.mean(f1_values), np.std(f1_values)

print(f"Arquivo: {global_file_path}")

print(f"Qtde tópicos: {global_n_topics}")

print(f"Qtde rotulados: {global_n_labeled}")


print(f"Média e desvio padrão da acurácia: {accuracy_mean}, {accuracy_std}")
print(
    f"Média e desvio padrão da precisão: {precision_mean}, {precision_std}")
print(f"Média e desvio padrão do recall: {recall_mean}, {recall_std}")
print(f"Média e desvio padrão do F1 score: {f1_mean}, {f1_std}")

for x in range(10):

    # print(f"Resultado nº {x+1}")

    # nmf_with_update(file_path, n_topics, n_labeled, max_iter, tol)
    nmf_with_update(data_folder / "tr11.mat.csv", 10, 20, 300, 1e-4)

# Calcula a média e o desvio-padrão das métricas
accuracy_mean, accuracy_std = np.mean(
    accuracy_values), np.std(accuracy_values)
precision_mean, precision_std = np.mean(
    precision_values), np.std(precision_values)
recall_mean, recall_std = np.mean(recall_values), np.std(recall_values)
f1_mean, f1_std = np.mean(f1_values), np.std(f1_values)

print(f"Arquivo: {global_file_path}")

print(f"Qtde tópicos: {global_n_topics}")

print(f"Qtde rotulados: {global_n_labeled}")

print(f"Média e desvio padrão da acurácia: {accuracy_mean}, {accuracy_std}")
print(
    f"Média e desvio padrão da precisão: {precision_mean}, {precision_std}")
print(f"Média e desvio padrão do recall: {recall_mean}, {recall_std}")
print(f"Média e desvio padrão do F1 score: {f1_mean}, {f1_std}")

for x in range(10):

    # print(f"Resultado nº {x+1}")

    # nmf_with_update(file_path, n_topics, n_labeled, max_iter, tol)
    nmf_with_update(data_folder / "tr11.mat.csv", 20, 20, 300, 1e-4)

# Calcula a média e o desvio-padrão das métricas
accuracy_mean, accuracy_std = np.mean(
    accuracy_values), np.std(accuracy_values)
precision_mean, precision_std = np.mean(
    precision_values), np.std(precision_values)
recall_mean, recall_std = np.mean(recall_values), np.std(recall_values)
f1_mean, f1_std = np.mean(f1_values), np.std(f1_values)

print(f"Arquivo: {global_file_path}")

print(f"Qtde tópicos: {global_n_topics}")

print(f"Qtde rotulados: {global_n_labeled}")

print(f"Média e desvio padrão da acurácia: {accuracy_mean}, {accuracy_std}")
print(
    f"Média e desvio padrão da precisão: {precision_mean}, {precision_std}")
print(f"Média e desvio padrão do recall: {recall_mean}, {recall_std}")
print(f"Média e desvio padrão do F1 score: {f1_mean}, {f1_std}")

for x in range(10):

    # print(f"Resultado nº {x+1}")

    # nmf_with_update(file_path, n_topics, n_labeled, max_iter, tol)
    nmf_with_update(data_folder / "tr11.mat.csv", 50, 20, 300, 1e-4)

# Calcula a média e o desvio-padrão das métricas
accuracy_mean, accuracy_std = np.mean(
    accuracy_values), np.std(accuracy_values)
precision_mean, precision_std = np.mean(
    precision_values), np.std(precision_values)
recall_mean, recall_std = np.mean(recall_values), np.std(recall_values)
f1_mean, f1_std = np.mean(f1_values), np.std(f1_values)

print(f"Arquivo: {global_file_path}")

print(f"Qtde tópicos: {global_n_topics}")

print(f"Qtde rotulados: {global_n_labeled}")

print(f"Média e desvio padrão da acurácia: {accuracy_mean}, {accuracy_std}")
print(
    f"Média e desvio padrão da precisão: {precision_mean}, {precision_std}")
print(f"Média e desvio padrão do recall: {recall_mean}, {recall_std}")
print(f"Média e desvio padrão do F1 score: {f1_mean}, {f1_std}")

############################################################################


for x in range(10):

    # print(f"Resultado nº {x+1}")

    # nmf_with_update(file_path, n_topics, n_labeled, max_iter, tol)
    nmf_with_update(data_folder / "tr11.mat.csv", 5, 30, 300, 1e-4)

# Calcula a média e o desvio-padrão das métricas
accuracy_mean, accuracy_std = np.mean(
    accuracy_values), np.std(accuracy_values)
precision_mean, precision_std = np.mean(
    precision_values), np.std(precision_values)
recall_mean, recall_std = np.mean(recall_values), np.std(recall_values)
f1_mean, f1_std = np.mean(f1_values), np.std(f1_values)

print(f"Arquivo: {global_file_path}")

print(f"Qtde tópicos: {global_n_topics}")

print(f"Qtde rotulados: {global_n_labeled}")


print(f"Média e desvio padrão da acurácia: {accuracy_mean}, {accuracy_std}")
print(
    f"Média e desvio padrão da precisão: {precision_mean}, {precision_std}")
print(f"Média e desvio padrão do recall: {recall_mean}, {recall_std}")
print(f"Média e desvio padrão do F1 score: {f1_mean}, {f1_std}")

for x in range(10):

    # print(f"Resultado nº {x+1}")

    # nmf_with_update(file_path, n_topics, n_labeled, max_iter, tol)
    nmf_with_update(data_folder / "tr11.mat.csv", 10, 30, 300, 1e-4)

# Calcula a média e o desvio-padrão das métricas
accuracy_mean, accuracy_std = np.mean(
    accuracy_values), np.std(accuracy_values)
precision_mean, precision_std = np.mean(
    precision_values), np.std(precision_values)
recall_mean, recall_std = np.mean(recall_values), np.std(recall_values)
f1_mean, f1_std = np.mean(f1_values), np.std(f1_values)

print(f"Arquivo: {global_file_path}")

print(f"Qtde tópicos: {global_n_topics}")

print(f"Qtde rotulados: {global_n_labeled}")

print(f"Média e desvio padrão da acurácia: {accuracy_mean}, {accuracy_std}")
print(
    f"Média e desvio padrão da precisão: {precision_mean}, {precision_std}")
print(f"Média e desvio padrão do recall: {recall_mean}, {recall_std}")
print(f"Média e desvio padrão do F1 score: {f1_mean}, {f1_std}")

for x in range(10):

    # print(f"Resultado nº {x+1}")

    # nmf_with_update(file_path, n_topics, n_labeled, max_iter, tol)
    nmf_with_update(data_folder / "tr11.mat.csv", 20, 30, 300, 1e-4)

# Calcula a média e o desvio-padrão das métricas
accuracy_mean, accuracy_std = np.mean(
    accuracy_values), np.std(accuracy_values)
precision_mean, precision_std = np.mean(
    precision_values), np.std(precision_values)
recall_mean, recall_std = np.mean(recall_values), np.std(recall_values)
f1_mean, f1_std = np.mean(f1_values), np.std(f1_values)

print(f"Arquivo: {global_file_path}")

print(f"Qtde tópicos: {global_n_topics}")

print(f"Qtde rotulados: {global_n_labeled}")

print(f"Média e desvio padrão da acurácia: {accuracy_mean}, {accuracy_std}")
print(
    f"Média e desvio padrão da precisão: {precision_mean}, {precision_std}")
print(f"Média e desvio padrão do recall: {recall_mean}, {recall_std}")
print(f"Média e desvio padrão do F1 score: {f1_mean}, {f1_std}")

for x in range(10):

    # print(f"Resultado nº {x+1}")

    # nmf_with_update(file_path, n_topics, n_labeled, max_iter, tol)
    nmf_with_update(data_folder / "tr11.mat.csv", 50, 30, 300, 1e-4)

# Calcula a média e o desvio-padrão das métricas
accuracy_mean, accuracy_std = np.mean(
    accuracy_values), np.std(accuracy_values)
precision_mean, precision_std = np.mean(
    precision_values), np.std(precision_values)
recall_mean, recall_std = np.mean(recall_values), np.std(recall_values)
f1_mean, f1_std = np.mean(f1_values), np.std(f1_values)

print(f"Arquivo: {global_file_path}")

print(f"Qtde tópicos: {global_n_topics}")

print(f"Qtde rotulados: {global_n_labeled}")

print(f"Média e desvio padrão da acurácia: {accuracy_mean}, {accuracy_std}")
print(
    f"Média e desvio padrão da precisão: {precision_mean}, {precision_std}")
print(f"Média e desvio padrão do recall: {recall_mean}, {recall_std}")
print(f"Média e desvio padrão do F1 score: {f1_mean}, {f1_std}")
