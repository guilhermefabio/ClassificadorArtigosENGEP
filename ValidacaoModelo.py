import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Carregar o arquivo Excel em um dataframe
df = pd.read_excel("areas_pesquisa_com_CAPES.xlsx")

# Separar as features (X) e o target (y)
X = df["PALAVRAS_CHAVE"]
y = df['Area_CAPES']

# Criar um vetorizador para converter as palavras-chave em vetores numéricos
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Criar e treinar o modelo Naive Bayes com validação cruzada
model = MultinomialNB()
y_pred = cross_val_predict(model, X_vectorized, y, cv=5)

# Calcular as diferenças entre as previsões e os valores reais
differences = (y_pred != y)

# Obter os índices dos dados que causam sobreajuste
overfitting_indices = np.where(differences)[0]

# Remover os dados que causam sobreajuste do dataframe original
df_limpo = df.drop(overfitting_indices)

# Salvar o novo dataframe em um arquivo Excel
df_limpo.to_excel("dados_limpos.xlsx", index=False)

# Separar as features (X_limpo) e o target (y_limpo) dos dados limpos
X_limpo = df_limpo["PALAVRAS_CHAVE"]
y_limpo = df_limpo['Area_CAPES']

# Criar um vetorizador para os dados limpos
X_limpo_vectorized = vectorizer.transform(X_limpo)

# Criar e treinar o modelo Naive Bayes com os dados limpos
model_limpo = MultinomialNB()
model_limpo.fit(X_limpo_vectorized, y_limpo)

# Realizar previsões no conjunto de teste
y_pred_limpo = model_limpo.predict(X_limpo_vectorized)


# Calcular as métricas para o modelo original
accuracy_original = accuracy_score(y, y_pred)
precision_original = precision_score(y, y_pred, average='weighted')
recall_original = recall_score(y, y_pred, average='weighted')
f1_score_original = f1_score(y, y_pred, average='weighted')
confusion_mtx_original = confusion_matrix(y, y_pred)

# Calcular as métricas para o modelo limpo
accuracy_limpo = accuracy_score(y_limpo, y_pred_limpo)
precision_limpo = precision_score(y_limpo, y_pred_limpo, average='weighted')
recall_limpo = recall_score(y_limpo, y_pred_limpo, average='weighted')
f1_score_limpo = f1_score(y_limpo, y_pred_limpo, average='weighted')
confusion_mtx_limpo = confusion_matrix(y_limpo, y_pred_limpo)

# Configurar os nomes das classes para a matriz de confusão
class_names = np.unique(np.concatenate((y, y_limpo)))
# Configurar o tamanho da figura e das subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plotar a matriz de confusão para o modelo original
axs[0].imshow(confusion_mtx_original, interpolation='nearest', cmap=plt.cm.Blues)
thresh_original = confusion_mtx_original.max() / 2.0
for i in range(confusion_mtx_original.shape[0]):
    for j in range(confusion_mtx_original.shape[1]):
        axs[0].text(j, i, confusion_mtx_original[i, j],
                    horizontalalignment="center",
                    color="white" if confusion_mtx_original[i, j] > thresh_original else "black",
                    fontsize=8)
axs[0].set_title('Matriz de Confusão (Modelo Original)')
axs[0].set_xticks(np.arange(len(class_names)))
axs[0].set_yticks(np.arange(len(class_names)))
axs[0].set_xticklabels(class_names, rotation=90, fontsize=8)
axs[0].set_yticklabels(class_names, fontsize=8)

# Plotar a matriz de confusão para o modelo limpo
axs[1].imshow(confusion_mtx_limpo, interpolation='nearest', cmap=plt.cm.Blues)
thresh_limpo = confusion_mtx_limpo.max() / 2.0
for i in range(confusion_mtx_limpo.shape[0]):
    for j in range(confusion_mtx_limpo.shape[1]):
        axs[1].text(j, i, confusion_mtx_limpo[i, j],
                    horizontalalignment="center",
                    color="white" if confusion_mtx_limpo[i, j] > thresh_limpo else "black",
                    fontsize=8)
axs[1].set_title('Matriz de Confusão (Modelo Limpo)')
axs[1].set_xticks(np.arange(len(class_names)))
axs[1].set_yticks(np.arange(len(class_names)))
axs[1].set_xticklabels(class_names, rotation=90, fontsize=8)
axs[1].set_yticklabels(class_names, fontsize=8)

# Ajustar o espaçamento entre as subplots
plt.tight_layout()

# Exibir a figura
plt.show()


# Imprimir os indicadores de desempenho para o modelo original
print("Indicadores de desempenho do modelo original:")
print("Acurácia:", accuracy_original)
print("Precisão:", precision_original)
print("Recall:", recall_original)
print("F1-score:", f1_score_original)

# Imprimir os indicadores de desempenho para o modelo limpo
print("\nIndicadores de desempenho do modelo limpo:")
print("Acurácia:", accuracy_limpo)
print("Precisão:", precision_limpo)
print("Recall:", recall_limpo)
print("F1-score:", f1_score_limpo)

