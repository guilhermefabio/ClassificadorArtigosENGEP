import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import re


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer




# Carregar o arquivo Excel em um dataframe
df = pd.read_excel("areas_pesquisa_com_CAPES.xlsx")

#df = pd.read_excel("novo_df_palavras_retiradas.xlsx")

X = df["PALAVRAS_CHAVE"]

print(X)
y =df['Area_CAPES']
print(y)
# Criar o objeto CountVectorizer
vectorizer = CountVectorizer()

# Transformar as palavras-chave em um vetor de recursos
X = vectorizer.fit_transform(X)

# Criar o objeto MultinomialNB
clf = MultinomialNB()

# Treinar o classificador
clf.fit(X, y)


# Fazer predições com novos dados
new_data = ['logistica, reciclagem, logistica reversa']
new_data = vectorizer.transform(new_data)
predictions = clf.predict(new_data)

# Imprimir as predições
print(predictions)

#separando os dados em variáveis de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

#criando o modelo de Naive Bayes Multinomial
model = MultinomialNB()
model.fit(X_train, y_train)

#realizando as previsões
y_pred = model.predict(X_test)

#calculando os indicadores de confiabilidade do modelo
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

#imprimindo os indicadores de confiabilidade do modelo
print('Acurácia:', accuracy)
print('Precisão:', precision)
print('Recall:', recall)
print('F1 Score:', f1)





import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Separar os dados em variáveis de treino e teste
X_train, X_test, y_train, y_test = train_test_split(df["PALAVRAS_CHAVE"], df["Area_CAPES"], test_size=0.2, random_state=42)

# Criar o objeto CountVectorizer
vectorizer = CountVectorizer()

# Ajustar o CountVectorizer aos dados de treinamento
X_train = vectorizer.fit_transform(X_train)

# Usar o CountVectorizer para transformar os dados de teste
X_test = vectorizer.transform(X_test)

# Criar o objeto MultinomialNB
clf = MultinomialNB()

# Treinar o classificador
clf.fit(X_train, y_train)

# Fazer predições com os dados de teste
y_pred = clf.predict(X_test)

# Calcular a matriz de confusão
cm = confusion_matrix(y_test, y_pred)

# Plotar a matriz de confusão em um heatmap
sns.heatmap(cm, annot=True, cmap="Blues")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.show()


# Calcular as palavras-chave mais ofensivas
coef = clf.coef_
keywords = vectorizer.get_feature_names()
df_keywords = pd.DataFrame({"keywords": keywords, "coef": coef[0]})
df_keywords = df_keywords.sort_values("coef", ascending=False).head(30)

# Plotar um gráfico de barras das palavras-chave ofensivas
plt.figure(figsize=(15,10))
sns.barplot(data=df_keywords.head(30), x="keywords", y="coef")
plt.xticks(rotation=90)
plt.xlabel("Palavras-chave")
plt.ylabel("Coeficiente")
plt.title("Palavras-chave mais ofensivas")
plt.show()

# Remover as palavras-chave ofensivas do dataframe
offensive_keywords = df_keywords["keywords"].tolist()
df_cleaned = df[~df["PALAVRAS_CHAVE"].str.contains("|".join(offensive_keywords), case=False, regex=True)]

# Salvar o dataframe limpo como um novo arquivo Excel
df_cleaned.to_excel("novo_df_palavras_retiradas.xlsx", index=False)




import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Carregar o arquivo Excel em um dataframe
df = pd.read_excel("areas_pesquisa_com_CAPES.xlsx")

X = df["PALAVRAS_CHAVE"]
y = df['Area_CAPES']

# Criar o objeto CountVectorizer
vectorizer = CountVectorizer()

# Transformar as palavras-chave em um vetor de recursos
X = vectorizer.fit_transform(X)

# Criar o objeto MultinomialNB
clf = MultinomialNB()

# Treinar o classificador
clf.fit(X, y)

# Fazer previsões com novos dados
new_data = input("Insira os dados para previsão: ")
new_data = vectorizer.transform([new_data])
predictions = clf.predict(new_data)

# Imprimir as predições
print(predictions)


'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir os modelos de machine learning a serem testados
models = [
    ('Logistic Regression', LogisticRegression(random_state=42)),
    ('Decision Tree', DecisionTreeClassifier(random_state=42)),
    ('Random Forest', RandomForestClassifier(random_state=42))
]

# Treinar os modelos com o conjunto de treinamento e fazer previsões com o conjunto de teste
results = {}
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-score': f1_score(y_test, y_pred)
    }

# Plotar os resultados em gráficos
for metric in ['Accuracy', 'Precision', 'Recall', 'F1-score']:
    plt.figure()
    plt.bar(results.keys(), [result[metric] for result in results.values()])
    plt.title(metric)
    plt.show()
'''