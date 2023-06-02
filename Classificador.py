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
Modelo = MultinomialNB()

# Treinar o classificador
Modelo.fit(X, y)

# Fazer previsões com novos dados
novos_dados = input("Insira os dados para previsão: ")
novos_dados = vectorizer.transform([novos_dados])
predicao = Modelo.predict(novos_dados)

# Calcular a porcentagem de acerto da previsão
probabiliades = Modelo.predict_proba(novos_dados)[0]
probabiliades_max = max(probabiliades)
accuracy = probabiliades_max * 100

# Imprimir a previsão
print("Previsão:", predicao[0])

# Imprimir a porcentagem de acerto
print("Porcentagem de acerto: {:.2f}%".format(accuracy))

# Verificar se a porcentagem de acerto é maior que 91%
if accuracy > 91:
    # Criar um dataframe com os dados da previsão
    df_pred = pd.DataFrame({"Dados": novos_dados, "Previsão": predicao[0], "Porcentagem de Acerto": accuracy}, index=[0])
    
    # Anexar o dataframe de previsão ao arquivo Excel de dados
    with pd.ExcelWriter("dados_limpos.xlsx", mode="a") as writer:
        df_pred.to_excel(writer, sheet_name="Previsões", index=False)
