# Importações necessárias
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelBinarizer


class DeepNMF:
    def __init__(self, k, n_labeled, max_iter=100):
        self.k = k  # Número de tópicos
        self.n_labeled = n_labeled  # Número de documentos rotulados na classe positiva
        self.positive_class = None  # Classe considerada positiva
        self.max_iter = max_iter  # Número máximo de iterações
        self.W = None  # Matriz documento-tópico
        self.H = None  # Matriz tópico-palavra
        self.V = None  # Matriz documento-palavra (TF-IDF)
        self.document_classes = None  # Classes dos documentos

    def load_data(self, filepath):
        # Carrega o dataset
        df = pd.read_csv(filepath)
        self.document_classes = df['Class'].values

        # Seleciona aleatoriamente uma classe positiva do dataset
        self.positive_class = np.random.choice(
            np.unique(self.document_classes))

        # Transforma a coluna de texto em uma matriz TF-IDF
        vectorizer = TfidfVectorizer(max_features=1000)
        self.V = vectorizer.fit_transform(df['Text']).toarray()

        # Inicializa as matrizes W e H
        self.W = np.random.rand(self.V.shape[0], self.k)
        self.H = np.random.rand(self.k, self.V.shape[1])

    def update_matrices(self):
        # Atualiza as matrizes W e H usando o Multiplicative Update
        for i in range(self.max_iter):
            # Atualização de H
            H_numerador = self.W.T.dot(self.V)
            H_denominador = self.W.T.dot(self.W).dot(
                self.H) + 1e-4  # Evita divisão por zero
            self.H = self.H * (H_numerador / H_denominador)

            # Atualização de W
            W_numerador = self.V.dot(self.H.T)
            W_denominador = self.W.dot(self.H).dot(
                self.H.T) + 1e-4  # Evita divisão por zero
            self.W = self.W * (W_numerador / W_denominador)

            # Aplicar restrições para documentos rotulados
            self.apply_constraints()

            # Cálculo do erro
            error = np.linalg.norm(self.V - self.W.dot(self.H), 'fro')
            print(f"Iteração {i+1}, Erro de Aproximação: {error}")

            # Cálculo das métricas de avaliação
            self.evaluate()

# Nota: As chamadas de métodos foram comentadas para evitar execução neste estágio.
# deep_nmf = DeepNMF(k=10, n_labeled=100, positive_class='Theory')
# deep_nmf.load_data('dataset.csv')
# deep_nmf.update_matrices()

    def apply_constraints(self):
        # Identifica os índices dos documentos rotulados na classe positiva
        labeled_indices = np.where(self.document_classes == self.positive_class)[
            0][:self.n_labeled]

    # Para cada documento rotulado, ajusta os valores na matriz W
        for idx in labeled_indices:
            self.W[idx, :] = 0.001  # Define todos os valores como 0.001
            # Define o primeiro valor como o maior valor de W
            self.W[idx, 0] = np.max(self.W)

    def evaluate(self):
        # Converte as predições das matrizes W para classes preditas
        predicted_topics = np.argmax(self.W, axis=1)
        predicted_classes = np.array(
            [self.positive_class if topic == 0 else "Other" for topic in predicted_topics])

        # Calcula as métricas de avaliação usando as classes reais e as classes preditas
        accuracy = accuracy_score(self.document_classes, predicted_classes)
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.document_classes, predicted_classes, average='macro', pos_label=self.positive_class, zero_division=0)

        print(
            f"Acurácia: {accuracy:.4f}, Precisão: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")


# Integração dos métodos na classe
# Note: As funções de avaliação dependem de uma interpretação binária das classes, o que pode requerer ajustes
# na lógica de classificação e na transformação das classes para o cálculo das métricas.


# Supondo que a classe DeepNMF já esteja definida como discutido anteriormente.

if __name__ == "__main__":
    # Define os parâmetros para a instância DeepNMF
    k = 10  # Número de tópicos
    n_labeled = 50  # Número de documentos rotulados na classe positiva
    # Nome da classe considerada positiva
    max_iter = 50  # Número máximo de iterações para o processo de atualização

    # Instancia o modelo DeepNMF com os parâmetros definidos
    deep_nmf = DeepNMF(k=k, n_labeled=n_labeled, max_iter=max_iter)

    # Carrega os dados do arquivo CSV
    deep_nmf.load_data(
        'C:/Users/lucsa/Dropbox/Data Science/Mestrado UNB/Dissertação/Experimentos/Testes/Deep NMF/Datasets/CSTR.csv')

    # Executa o processo de fatoração e atualiza as matrizes W e H
    deep_nmf.update_matrices()

    # Nota: O método `update_matrices` já inclui a aplicação das restrições e o cálculo das métricas de avaliação
    # a cada iteração, conforme implementado nos métodos `apply_constraints` e `evaluate`.
