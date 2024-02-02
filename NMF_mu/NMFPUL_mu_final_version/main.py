from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from random import choice
import os
from NMF_mu.NMFPUL_mu_final_version import nmf_with_update

# Inicializa listas para armazenar os valores das métricas
accuracy_values = []
precision_values = []
recall_values = []
f1_values = []

data_folder = Path(
    "C:/Users/lucsa/Dropbox/Data Science/Mestrado UNB/Dissertação/Experimentos/Testes/NMF/Datasets/")

for x in range(10):

    print(f"Resultado nº {x+1}")

    # nmf_with_update(file_path, n_topics, n_labeled, max_iter, tol)
    nmf_with_update(data_folder / "CSTR.csv", 50, 1, 300, 1e-4)

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
