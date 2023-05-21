#teste_matriz
import pandas as pd

df = pd.read_csv("C:/Users/Guilherme Vieira/Desktop/TCC/Busca_dados_ENEGEP/excel_arquivos/matriz_treino_csv.csv")
print(df.head())
pd.plotting.scatter_matrix(df, alpha=0.2)

