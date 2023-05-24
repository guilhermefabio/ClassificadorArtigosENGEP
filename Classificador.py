import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Carregar o arquivo Excel em um dataframe
df = pd.read_excel("dados_limpos.xlsx")

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

# Calcular a porcentagem de acerto da previsão
probabilities = clf.predict_proba(new_data)[0]
max_probability = max(probabilities)
accuracy = max_probability * 100

# Imprimir a previsão
print("Previsão:", predictions[0])

# Imprimir a porcentagem de acerto
print("Porcentagem de acerto: {:.2f}%".format(accuracy))

# Verificar se a porcentagem de acerto é maior que 91%
if accuracy > 91:
    # Criar um dataframe com os dados da previsão
    df_pred = pd.DataFrame({"Dados": new_data, "Previsão": predictions[0], "Porcentagem de Acerto": accuracy}, index=[0])
    
    # Anexar o dataframe de previsão ao arquivo Excel de dados
    with pd.ExcelWriter("dados_limpos.xlsx", mode="a") as writer:
        df_pred.to_excel(writer, sheet_name="Previsões", index=False)
