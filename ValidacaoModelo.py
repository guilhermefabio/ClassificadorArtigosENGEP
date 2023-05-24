import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def load_data(file_path):
    # Carregar o arquivo Excel em um dataframe
    df = pd.read_excel(file_path)
    
    # Separar as features (X) e o target (y)
    X = df["PALAVRAS_CHAVE"]
    y = df['Area_CAPES']
    
    return X, y

def preprocess_data(X, y):
    # Criar um vetorizador para converter as palavras-chave em vetores numéricos
    vectorizer = CountVectorizer()
    X_vectorized = vectorizer.fit_transform(X)
    
    return X_vectorized, vectorizer, y

def train_model(X, y):
    # Criar e treinar o modelo Naive Bayes com validação cruzada
    model = MultinomialNB()
    y_pred = cross_val_predict(model, X, y, cv=5)
    
    return model, y_pred

def evaluate_model(y_true, y_pred):
    # Calcular as métricas de desempenho
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    confusion_mtx = confusion_matrix(y_true, y_pred)
    classification_rep = classification_report(y_true, y_pred)
    
    return accuracy, precision, recall, f1, confusion_mtx, classification_rep

def plot_confusion_matrix(confusion_mtx, class_names, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mtx, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(title)
    plt.xlabel('Classe Preditas')
    plt.ylabel('Classes Reais')
    plt.xticks(np.arange(len(class_names)) + 0.5, class_names, rotation=90)
    plt.yticks(np.arange(len(class_names)) + 0.5, class_names, rotation=0)
    plt.show()


def compare_models(model1, y_pred1, model2, y_pred2, class_names):
    # Comparar as métricas e plotar as matrizes de confusão
    metrics1 = evaluate_model(y, y_pred1)
    metrics2 = evaluate_model(y_limpo, y_pred2)

    metric_names = ['Acurácia', 'Precisão', 'Recall', 'F1-score']
    values1 = metrics1[:4]
    values2 = metrics2[:4]

    # Plotar gráfico comparativo das métricas
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    bars1 = ax1.bar(metric_names, values1, label='Modelo Original', alpha=0.7)
    ax1.set_xlabel('Métricas')
    ax1.set_ylabel('Valor')
    ax1.set_title('Modelo Original')
    ax1.legend()

    # Adicionar anotações nos gráficos de barras
    for bar1 in bars1:
        height1 = bar1.get_height()
        ax1.text(bar1.get_x() + bar1.get_width() / 2, height1, round(height1, 2),
                 ha='center', va='bottom')

    bars2 = ax2.bar(metric_names, values2, label='Modelo Limpo', alpha=0.7)
    ax2.set_xlabel('Métricas')
    ax2.set_ylabel('Valor')
    ax2.set_title('Modelo Limpo')
    ax2.legend()

    # Adicionar anotações nos gráficos de barras
    for bar2 in bars2:
        height2 = bar2.get_height()
        ax2.text(bar2.get_x() + bar2.get_width() / 2, height2, round(height2, 2),
                 ha='center', va='bottom')

    plt.suptitle('Comparação das Métricas entre o Modelo Original e o Modelo Limpo')
    plt.tight_layout()
    plt.show()

    # Plotar as matrizes de confusão para o modelo original e o modelo limpo
    plot_confusion_matrix(metrics1[4], class_names, 'Matriz de Confusão - Modelo Original')
    plot_confusion_matrix(metrics2[4], class_names, 'Matriz de Confusão - Modelo Limpo')

    # Imprimir os relatórios de classificação para o modelo original e o modelo limpo
    print("Relatório de Classificação - Modelo Original:")
    print(metrics1[5])
    print("\nRelatório de Classificação - Modelo Limpo:")
    print(metrics2[5])
    


def plot_confusion_matrix(confusion_matrix, class_names, title):
    plt.figure(figsize=(10, 10))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, [str(i) for i in range(len(class_names))], rotation=0, ha='center')  # Apenas números no eixo x
    plt.yticks(tick_marks, [str(i) + ' ' + name for i, name in enumerate(class_names)], va='center')  # Número + nome no eixo y

    # Adicionar valores dentro das células da matriz de confusão
    thresh = confusion_matrix.max() / 2.
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            plt.text(j, i, format(confusion_matrix[i, j], 'd'),
                     horizontalalignment="center",
                     verticalalignment="center",
                     color="white" if confusion_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Classe Verdadeira')
    plt.xlabel('Classe Predita')
    plt.gca().set_aspect('equal')  # Ajuste da proporção da imagem para evitar distorções
    plt.show()



def analyze_model_differences(X, y_true, y_pred1, y_pred2):
    # Calcular as diferenças entre as previsões dos modelos
    differences = (y_pred1 != y_pred2)
    
    # Obter os índices dos exemplos em que os modelos diferem
    differing_indices = np.where(differences)[0]
    
    # Examinar os exemplos em que os modelos diferem
    for idx in differing_indices:
        print("Exemplo:", X[idx])
        print("Classe real:", y_true[idx])
        print("Previsão do Modelo Original:", y_pred1[idx])
        print("Previsão do Modelo Limpo:", y_pred2[idx])
        print("----------------------------------------")

def print_confusion_matrix(confusion_mtx, class_names):
    num_classes = len(class_names)

    # Imprimir cabeçalho das classes preditas
    header = "\t".join(class_names)
    print("Pred / True\t" + header)

    # Imprimir linhas da matriz de confusão
    for i in range(num_classes):
        row = class_names[i] + "\t"
        for j in range(num_classes):
            cell = str(confusion_mtx[i][j])
            row += cell + "\t"
        print(row)

# Carregar os dados
X, y = load_data("areas_pesquisa_com_CAPES.xlsx")

# Pré-processar os dados
X_vectorized, vectorizer, y = preprocess_data(X, y)

# Treinar o modelo original
model_original, y_pred_original = train_model(X_vectorized, y)

# Remover os dados que causam sobreajuste
overfitting_indices = np.where(y_pred_original != y)[0]
X_limpo = X.drop(overfitting_indices)
y_limpo = y.drop(overfitting_indices)

# Pré-processar os dados limpos
X_limpo_vectorized = vectorizer.transform(X_limpo)

# Treinar o modelo limpo
model_limpo, y_pred_limpo = train_model(X_limpo_vectorized, y_limpo)

# Comparar os modelos
class_names = np.unique(np.concatenate((y, y_limpo)))
compare_models(model_original, y_pred_original, model_limpo, y_pred_limpo, class_names)

# Analisar as diferenças entre os modelos
analyze_model_differences(X, y, y_pred_original, y_pred_limpo)

# Criar um dataframe com os dados limpos
df_limpo = pd.DataFrame({'PALAVRAS_CHAVE': X_limpo, 'Area_CAPES': y_limpo})

# Salvar o dataframe de dados limpos em um arquivo Excel
df_limpo.to_excel("dados_limpos.xlsx", index=False)
