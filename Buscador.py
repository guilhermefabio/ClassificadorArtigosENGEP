# Importação de bibliotecas necessárias
from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
import re
import requests
import os

# Função para dividir uma lista em pedaços menores
def chunks(lista, n):
    for i in range(0, len(lista), n):
        yield lista[i:i + n]

# Função para obter os dados brutos da página
def Get_Raw_Data(URL):
	url = URL
	html = urlopen(url).read()
	soup = BeautifulSoup(html, features="html.parser")
	
	# Remove as tags de script e estilo do HTML
	for script in soup(["script", "style"]):
		   script.extract()    
	
	# Extrai os dados do HTML
	text = soup.get_text()
	
	# Separa as linhas do texto
	lines = (line.strip() for line in text.splitlines()) 
	
	# Separa frases em branco
	chunks = (phrase.strip() for line in lines for phrase in line.split("	"))
	
	# Insere uma palavra controle para corte
	text = '"ENEGEP"'.join(chunk for chunk in chunks if chunk) 
	
	# Divide a lista de dados pelo controle inserido anteriormente
	text = text.split('ENEGEP') 
	
	# Retorna a lista com os dados brutos da página
	a_list = text
	return a_list, text

# Função para extrair todo o conteúdo HTML da página e salvá-lo em um arquivo de texto
def Extract_Full_HTML_Page(textfile, a_list):
	for element in a_list:
		textfile.write(str(element) + '\n')
	textfile.close()

# Função para pré-processar os dados brutos da página
def Pre_Process_Data(text, ANOS):
	newdata = []
	data = []
	y = 0
	z = 1000000
	
	# Percorre a lista de dados brutos e separa as informações relevantes
	for i in text:
		if i == '"Resultado da Pesquisa"':
			z = y + 1
		if i == '"  ':
			data.append(newdata)
		if i.find('Página (Page) :')==True:
			break
		if y > z:
			newdata.append(i)
		y = y + 1
	n = 1

	newnewdata = []
	for i in newdata:
		if i != '"AUTORES:"' and i !='"' and i!= ' ':
			newnewdata.append(i)

	titulos = ['0', '1', '2', '3', '4', '5']

	for item in newnewdata:
		x = re.search('enegep', item)
		if x:
			newnewdata.remove(item)

	# Separa as informações relevantes em sublistas de tamanho 5
	newnewdata = [newnewdata[i:i + n] for i in range(0, len(newdata), n)]

	# Separa as sublistas pelo item '"'
	newnewdata = [x[:x.index('"')] if '"' in x else x for x in newnewdata] 
	
	# Limpa itens em branco
	newnewdata = [i for i in newnewdata if i]

	data_set = []
	for i in range(len(newnewdata)):
		nome = ' '.join([str(item) for item in newnewdata[i]])
		if re.search(str(ANOS[a]) + "_T", nome):  
			new_list = newnewdata[i:i + 6]
			data_set.append(new_list)
	return data_set

# ELEMENTOS DE BUSCA
ANOS = [2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021]
range_inicial = 57

# Criação de DataFrames vazios
df2 = pd.DataFrame()
df3 = pd.DataFrame()

# Loop para buscar os dados de diferentes anos e áreas temáticas
for a in range(len(ANOS)):
	++a
	for key in range(range_inicial, 363):
		# Constrói a URL de busca
		URL = 'http://www.abepro.org.br/publicacoes/index.asp?pesq=ok&ano=' + str(ANOS[a]) + '&area=' + str(key) + '&pchave=&autor='
		
		# Obtém os dados brutos da página
		a_list, text = Get_Raw_Data(URL)
		print("\n","Busca de dados da pagina" +  URL + "  Concluida.......")
		
		# Extrai o conteúdo HTML completo da página e o salva em um arquivo
		textfile = open(r"html_arquivos/dados_pagina"+"_"+str(key)+"_"+str(ANOS[a])+".txt", "w",encoding="utf-8")
		Extract_Full_HTML_Page(textfile, a_list)
		print("\n","HTML Extraido e Salvo.......")
		
		# Pré-processa os dados brutos da página
		data_set = Pre_Process_Data(text, ANOS)
		print("\n","Pré processamento executado......")

		# Criação de DataFrame com os dados processados
		df = pd.DataFrame(data_set, columns=['ID', 'TITULO', 'AREA', 'AUTORES', 'PALAVRAS_CHAVE', 'RESUMO'])
		
		# Verifica se o DataFrame possui mais de uma linha
		if len(df) > 1:
			df2 = df2.append(df)
			r = requests.get(URL)
			df_list = pd.read_html(r.text)
			df4 = df_list[2]
			df3 = df3.append(df_list[2])
		else:
			# Verifica se o limite de busca foi atingido para avançar para o próximo ano
			if key - range_inicial >= 11 and ANOS[a] != 2019:
				range_inicial = key
				break
		print(df)

# Imprime os DataFrames resultantes
print(df2)
print(df3)
# Salva os DataFrames em arquivos Excel
df3.to_excel("excel_arquivos/autores_total.xlsx")
df2.to_excel("excel_arquivos/artigos_total.xlsx")
