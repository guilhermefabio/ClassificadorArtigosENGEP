
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