import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Number of iterations
N = 10

# Initialize lists to store metrics for each iteration
accuracy_list = []
precision_list = []
recall_list = []
f1_list = []


for _ in range(N):

    # Load data
    df = pd.read_csv(
        'C:/Users/lucsa/Dropbox/Data Science/Mestrado UNB/Dissertação/Experimentos/Testes/NMF/Datasets/Reviews.csv')

    # Store original classes
    df['Original Class'] = df['Class']

    # Process text with TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['Text'])

    # Choose positive class
    positive_class = df['Class'].value_counts().idxmax()

    # Define percentage of positive examples that will be labeled
    percentage_labeled = 0.5

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
        df['Original Class'], final_predictions, average='macro', zero_division=1)
    recall = recall_score(df['Original Class'],
                          final_predictions, average='macro', zero_division=1)
    f1 = f1_score(df['Original Class'], final_predictions,
                  average='macro', zero_division=1)

    # Append metrics to lists
    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)

# Calculate mean and standard deviation of metrics
accuracy_mean = np.mean(accuracy_list)
precision_mean = np.mean(precision_list)
recall_mean = np.mean(recall_list)
f1_mean = np.mean(f1_list)

accuracy_std = np.std(accuracy_list)
precision_std = np.std(precision_list)
recall_std = np.std(recall_list)
f1_std = np.std(f1_list)

# Print results
print(f"Accuracy: {accuracy_mean} (+/- {accuracy_std})")
print(f"Precision: {precision_mean} (+/- {precision_std})")
print(f"Recall: {recall_mean} (+/- {recall_std})")
print(f"F1-score: {f1_mean} (+/- {f1_std})")
