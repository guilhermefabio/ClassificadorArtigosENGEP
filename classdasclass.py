import pandas as pd
from fuzzywuzzy import fuzz, process
import re

def preprocess_text(text):
    # Remover pontuações e caracteres especiais
    text = re.sub('[^a-zA-Z]', ' ', text)
    
    # Converter todas as letras para minúsculas
    text = text.lower()
    
    # Tokenizar o texto em palavras
    tokens = word_tokenize(text)
    
    # Remover stopwords
    stop_words = set(stopwords.words('portuguese'))  # ou 'english' para stopwords em inglês
    tokens = [word for word in tokens if word not in stop_words]
    
    # Aplicar stemming (opcional)
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    
    # Juntar as palavras novamente em uma única string
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text

# Carregar o arquivo Excel em um dataframe
df = pd.read_excel(r"excel_arquivos/artigos_total.xlsx")


# Selecionar as colunas "area" e "palavras-chave"
X = df["PALAVRAS_CHAVE"].apply(lambda x: re.sub('[,.&@;"\[\]\\\//0-9:]', '', x))
#y = df["AREA"].apply(lambda x: re.sub('[,.&@;"\[\]\\\//0-9:\t]', '', x))

dfAreas= df["AREA"].str.split("/", expand=True)
#print(dfAreas)
dfAreas[0] = dfAreas[0].str.replace('["\[\]\d.:]', "") #areas
dfAreas[1] = dfAreas[1].str.replace('["\[\]\d.:]', "") #areas detalhadas
print(dfAreas[0])

y = dfAreas[0].str.replace('[,.&@;"\[\]\\\//0-9:\t]',"")


df_concat = pd.concat([X, y], axis=1)
df_concat.to_excel('arquivo_limpo.xlsx', index=False)



# lendo a base com as áreas do CAPES
df_capes = pd.read_excel('capes_areas.xlsx')

# lendo a base com as áreas de pesquisa
df_pesquisa = pd.read_excel('arquivo_limpo.xlsx')

# criando uma lista com as áreas do CAPES
areas_capes = df_capes['CAPES'].tolist()

# criando uma lista com as áreas de pesquisa
areas_pesquisa = df_pesquisa[0].tolist()

# criando um dicionário para armazenar as correspondências encontradas
correspondencias = {}

# percorrendo cada área de pesquisa
for area_pesquisa in areas_pesquisa:
    # encontrando a melhor correspondência com as áreas do CAPES usando fuzzy matching
    melhor_correspondencia, score = process.extractOne(area_pesquisa, areas_capes, scorer=fuzz.token_sort_ratio)
    
    # adicionando a correspondência encontrada no dicionário
    correspondencias[area_pesquisa] = melhor_correspondencia

# criando uma nova coluna na base de pesquisa com as correspondências encontradas
df_pesquisa['Area_CAPES'] = df_pesquisa[0].map(correspondencias)
print(df_pesquisa['Area_CAPES'])
# salvando a base de pesquisa com as correspondências encontradas em um novo arquivo Excel
df_pesquisa.to_excel('areas_pesquisa_com_CAPES.xlsx', index=False)
