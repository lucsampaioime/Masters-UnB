import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Carregar dados do arquivo CSV
df = pd.read_csv(
    'C:/Users/lucsa/Dropbox/Data Science/Mestrado UNB/Dissertação/Experimentos/Testes/NMF/Datasets/CSTR.csv')

# Store original classes
df['Original Class'] = df['Class']

# Process text with TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Text'])

# Choose positive class
positive_class = np.random.choice(df['Class'].unique())

# Define percentage of positive examples that will be labeled
percentage_labeled = 0.2

# Set 'Class' to undefined for non-positive classes and for a subset of positive examples
df['Class'] = np.where((df['Class'] == positive_class) &
                       (np.random.rand(len(df)) > percentage_labeled),
                       positive_class,
                       'undefined')

# Apply NMF
# Number of unique classes in the original data
n_topics = df['Original Class'].nunique()
nmf = NMF(n_components=n_topics, solver='mu')
W = nmf.fit_transform(X)

# Train initial classifier
svm = SVC()
svm.fit(W, np.where(df['Class'] == positive_class, 1, 0))

# Get the indices of the unlabeled examples
unlabeled_indices = df[df['Class'] == 'undefined'].index

# Reevaluate unlabeled examples
unlabeled_predictions = svm.predict(W[unlabeled_indices])

# Separate unlabeled examples into positive and negative
df.loc[unlabeled_indices[unlabeled_predictions == 1], 'Class'] = positive_class
df.loc[unlabeled_indices[unlabeled_predictions == 0], 'Class'] = 'negative'

# Retrain classifier with newly labeled data
svm.fit(W, df['Class'])

# Classify all examples
final_predictions = svm.predict(W)

# Calculate metrics
accuracy = accuracy_score(df['Original Class'], final_predictions)
precision = precision_score(
    df['Original Class'], final_predictions, average='weighted')
recall = recall_score(df['Original Class'],
                      final_predictions, average='weighted')
f1 = f1_score(df['Original Class'], final_predictions, average='weighted')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")
