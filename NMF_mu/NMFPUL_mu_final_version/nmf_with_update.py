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

    # Determina qual tópico é o mais representativo para a classe positiva [usar uma das opções abaixo]
    # positive_class_index = np.argmax(np.sum(W[classes == 1], axis=0))
    positive_class_index = 0

    # Cria máscaras booleanas para os documentos rotulados e não rotulados
    labeled_mask = np.zeros(len(classes), dtype=bool)
    labeled_mask[labeled] = True
    unlabeled_mask = ~labeled_mask

    for n in range(max_iter):
        # Atualiza as matrizes W e H - Euclidian
        W *= (V @ H.T) / (W @ (H @ H.T) + np.finfo(float).eps)
        H *= (W.T @ V) / ((W.T @ W) @ H + np.finfo(float).eps)

        # Atualiza as matrizes W e H - KL
        # W *= (V @ H.T) / (W @ H @ H.T + 1e-10)
        # H *= (W.T @ V) / (W.T @ W @ H + 1e-10)

        # Aplique a função 'Supress'
        W[labeled, 0] = np.max(W)
        W[labeled, 1:] = 0.001

        # Calcula a norma euclidiana entre V e WH
        error = np.linalg.norm(V - W @ H)

        # Calcula a divergencia KL entre V e WH
        # error = np.sum(V * np.log(V / (W @ H + 1e-10)) - V + W @ H)

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
