from matplotlib import pyplot as plt
from matplotlib.pyplot import text
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import sklearn.decomposition as sc
import numpy as np
import matplotlib.ticker as mticker
import joblib
import torch
import torch.nn as nn
import random
from scipy.optimize import nnls
from random import choice
from pathlib import Path
import os
EPSILON = np.finfo(np.float32).eps

# Inicializa listas para armazenar os valores das métricas
accuracy_values = []
precision_values = []
recall_values = []
f1_values = []


class UnsuperLayer(nn.Module):
    """
    Multiplicative update with Frobenius norm
    This can fit L1, L2 regularization.
    """

    def __init__(self, comp, features, l_1, l_2):
        super(UnsuperLayer, self).__init__()
        # an affine operation: y = Wx +b
        self.l_1 = l_1
        self.l_2 = l_2
        self.fc1 = nn.Linear(comp, comp, bias=False)
        self.fc2 = nn.Linear(features, comp, bias=False)

    def forward(self, y, x):
        denominator = torch.add(self.fc1(y), self.l_2 * y + self.l_1 + EPSILON)
        numerator = self.fc2(x)
        delta = torch.div(numerator, denominator)
        return torch.mul(delta, y)


class UnsuperNet(nn.Module):
    """
    Class for a Regularized DNMF with varying layers number.
    Input:
        -n_layers = number of layers to construct the Net
        -comp = number of components for factorization
        -features = original features length for each sample vector(mutational sites)
    each layer is MU of Frobenius norm
    """

    def __init__(self, n_layers, comp, features, l_1=0, l_2=0):
        super(UnsuperNet, self).__init__()
        self.n_layers = n_layers
        self.deep_nmfs = nn.ModuleList(
            [UnsuperLayer(comp, features, l_1, l_2)
             for i in range(self.n_layers)]
        )

    def forward(self, h, x):
        # sequencing the layers and forward pass through the network.
        for i, l in enumerate(self.deep_nmfs):
            h = l(h, x)
        return h


def cost_tns(v, w, h, l_1=0, l_2=0):
    # util.cost_tns(data.v_train.tns,data.w.tns,data.h_train.tns)
    d = v - h.mm(w)
    return (
        0.5 * torch.pow(d, 2).sum()
        + l_1 * h.sum()
        + 0.5 * l_2 * torch.pow(h, 2).sum()
    )/h.shape[0]


def tensoring(X):
    # convert numpy array to torch tensor
    return torch.from_numpy(X).float()


def build_data(V, H):

    V_tns = torch.from_numpy(V.T).float()
    H_tns = torch.from_numpy(H.T).float()

    return V_tns, H_tns


def train_unsupervised(V_tns, H_tns, W_init_tns, num_layers, network_train_iterations, n_components, features, lr=0.0005, l_1=0, l_2=0, verbose=False, include_reg=True, positive_class_indices=None):
    """
    Train the unsupervised deep NMF model.

    Parameters:
    - W_init_tns: Initial weights for W (tensor format).
    - num_layers: Number of layers in the deep NMF network.
    - network_train_iterations: Number of iterations for training the network.
    - n_components: Number of components for factorization.
    - features: Original features length for each sample vector.
    - lr: Learning rate for the optimizer.
    - l_1, l_2: Regularization parameters.
    - verbose: If True, print loss information during training.
    - include_reg: Include regularization in the model if True.

    Returns:
    - deep_nmf: Trained deep NMF model.
    - dnmf_cost: List of training costs.
    - dnmf_w: Trained weights for W.
    - out: output matrix H (documents, classes)
    """

    # Build the architecture
    deep_nmf = UnsuperNet(num_layers, n_components, features, l_1, l_2) if include_reg else UnsuperNet(
        num_layers, n_components, features, 0, 0)

    # Initialize parameters
    dnmf_w = W_init_tns.clone()
    for w in deep_nmf.parameters():
        w.data.fill_(0.1)

    optimizerADAM = torch.optim.Adam(deep_nmf.parameters(), lr=lr)

    # Train the Network
    inputs = (H_tns, V_tns)
    dnmf_cost = []

    for i in range(network_train_iterations):
        out = deep_nmf(*inputs)
        # print(f" W: {dnmf_w.shape}, H: {out.shape}, V: {V_tns.shape}")
        loss = cost_tns(V_tns, dnmf_w, out, l_1, l_2)

        if verbose:
            print(i, loss.item())

        # Zero the gradients before running the backward pass
        optimizerADAM.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()

        # Perform a single optimization step (parameter update)
        optimizerADAM.step()

        # Keep weights positive after gradient descent
        for w in deep_nmf.parameters():
            w.data = w.clamp(min=0)

        h_out = torch.transpose(out.data, 0, 1)

        # Assuming out.data is of shape [samples, components]
        valid_indices = [
            i for i in positive_class_indices if i < out.data.shape[1]]

        # Modification: Set the highest value in H for documents belonging to the positive class
        if positive_class_indices is not None:
            highest_value = out.data.max().item()  # Find the highest value in H

            for i in valid_indices:
                # Assuming you're modifying the entire feature/component vector for each valid sample
                out.data[:, i] = highest_value

        # NNLS
        w_arrays = [nnls(out.data.numpy(), V_tns[:, f].numpy())[0]
                    for f in range(features)]
        nnls_w = np.stack(w_arrays, axis=-1)
        dnmf_w = torch.from_numpy(nnls_w).float()

        dnmf_cost.append(loss.item())

        # Test performance
        dnmf_cost.append(
            cost_tns(V_tns, dnmf_w, out, l_1, l_2).item())

    return deep_nmf, dnmf_cost, dnmf_w, out


def read_and_process_csv(file_path):
    # load the dataset
    with open(file_path, 'r') as f:
        lines = f.readlines()[1:]  # skips first line

    # Divide each line in text and class
    texts, classes = zip(*[line.rsplit(',', 1) for line in lines])

    # Convert the classes instances into numbers
    le = LabelEncoder()
    classes = le.fit_transform(classes)

    # Escolhe uma classe aleatória para ser a positiva, e transforma o restante em negativa
    positive_class = classes[0]
    # positive_class = choice(np.unique(classes))
    classes = np.where(classes == positive_class, 1, 0)

    # vectorizer = CountVectorizer()
    vectorizer = TfidfVectorizer()
    V = vectorizer.fit_transform(texts).toarray()

    return V, classes


def label_documents(classes, n_labeled):

    # Chooses a random class to be selected as the positive one and transform the others classes in negative
    # positive_class = classes[0]
    positive_class = choice(np.unique(classes))
    classes = np.where(classes == positive_class, 1, 0)

    # Label a n_labeled quantity of documents of the positive class
    positive_indexes = np.where(classes == 1)[0]
    labeled = positive_indexes[:n_labeled]
    unlabeled = positive_indexes[n_labeled:]

    # Create boolean masks for the labeled and unlabeled documents
    labeled_mask = np.zeros(len(classes), dtype=bool)
    labeled_mask[labeled] = True
    unlabeled_mask = ~labeled_mask

    return labeled_mask, unlabeled_mask, labeled, positive_class


def nmf_with_update(V, W, H, n_topics, n_labeled, max_iter, tol, classes):

    # Rotula uma quantidade n_labeled de documentos da classe positiva
    positive_indexes = np.where(classes == 1)[0]
    # np.random.shuffle(positive_indexes)
    labeled = positive_indexes[:n_labeled]
    unlabeled = positive_indexes[n_labeled:]

    # Inicializa as matrizes W e H
    # W = np.random.rand(V.shape[0], n_topics)
    # H = np.random.rand(n_topics, V.shape[1])

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
            classes[unlabeled_mask], preds[unlabeled_mask], average='weighted', zero_division=0)
        recall = recall_score(
            classes[unlabeled_mask], preds[unlabeled_mask], average='weighted', zero_division=0)
        f1 = f1_score(classes[unlabeled_mask],
                      preds[unlabeled_mask], average='weighted', zero_division=0)
        cm = confusion_matrix(classes[unlabeled_mask], preds[unlabeled_mask])

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


def main():

    # Path to the CSV file
    file_path = 'C:/Users/lucsa/Dropbox/Data Science/Mestrado UNB/Dissertação/Experimentos/Testes/Deep NMF/Datasets/CSTR.csv'
    V, classes = read_and_process_csv(file_path)

    # Defines the dimensions of the V matrix (samples x words)
    n_samples, n_features = V.shape
    n_components = 30  # This can be defined based on the problem needs

    V = V.transpose()

    # Initialize W and H with random values
    W = np.random.rand(n_features, n_components)
    H = np.random.rand(n_components, n_samples)

    # Prepare data (split and convert to tensors)
    V_tns, H_tns = build_data(V, H)

    # Convert W to tensor
    W_init_tns = torch.from_numpy(W.T).float()

    # Parameters for the training
    num_layers = 10
    network_train_iterations = 100
    lr = 0.001
    l_1 = 0.1
    l_2 = 0.1

    n_labeled = 30  # Or get this from user input.
    labeled_mask, unlabeled_mask, labeled, positive_class = label_documents(
        classes, n_labeled)

    # Train the model
    model, cost, dnmf_w, dnmf_h = train_unsupervised(
        V_tns, H_tns, W_init_tns, num_layers, network_train_iterations, n_components, n_features, lr, l_1, l_2, verbose=True, positive_class_indices=labeled)

    # Calculate error
    reconstructed_V = H_tns.mm(dnmf_w)
    error = torch.norm(V_tns - reconstructed_V, p='fro').item()
    print("Reconstruction Error:", error)

    final_H = dnmf_w.cpu().detach().numpy()
    final_W = dnmf_h.cpu().detach().numpy()

    # print(f"W: {final_W.shape}")
    # print(f"H: {final_H.shape}")
    # print(f"V: {V.shape}")

    V = V.transpose()

    for x in range(10):

        # nmf_with_update(matrix V,  matrix W, matriz H, n_topics, n_labeled, max_iter, tol, classes)
        nmf_with_update(V, final_W, final_H, 30, 30, 30, 1e-4, classes)

    # Calcula a média e o desvio-padrão das métricas
    accuracy_mean, accuracy_std = np.mean(
        accuracy_values), np.std(accuracy_values)
    precision_mean, precision_std = np.mean(
        precision_values), np.std(precision_values)
    recall_mean, recall_std = np.mean(recall_values), np.std(recall_values)
    f1_mean, f1_std = np.mean(f1_values), np.std(f1_values)

    print(f"Arquivo: {os.path.basename(file_path)}")

    print(f"Qtde tópicos: {global_n_topics}")

    print(f"Qtde rotulados: {global_n_labeled}")

    print(
        f"Média e desvio padrão da acurácia: {accuracy_mean}, {accuracy_std}")
    print(
        f"Média e desvio padrão da precisão: {precision_mean}, {precision_std}")
    print(f"Média e desvio padrão do recall: {recall_mean}, {recall_std}")
    print(f"Média e desvio padrão do F1 score: {f1_mean}, {f1_std}")


if __name__ == "__main__":
    main()
