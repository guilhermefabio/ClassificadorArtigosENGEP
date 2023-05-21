
import openai
openai.api_key = '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'

def obter_resposta(texto):
    resposta = openai.Completion.create(
        engine='text-davinci-003',
        prompt=texto,
        max_tokens=50,
        temperature=0.7,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    return resposta.choices[0].text.strip()

areas = [
    "GERÊNCIA DE PRODUÇÃO",
    "PLANEJAMENTO DE INSTALAÇOES INDUSTRIAIS",
    "PLANEJAMENTO, PROJETO E CONTROLE DE SIST. DE PRODUÇÃO",
    "HIGIENE E SEGURANÇA DO TRABALHO",
    "SUPRIMENTOS",
    "GARANTIA DE CONTROLE DE QUALIDADE",
    "PESQUISA OPERACIONAL",
    "PROCESSOS ESTOCÁSTICOS E TEORIAS DAS FILAS",
    "PROGRAMAÇÃO LINEAR, NÃO-LINEAR, MISTA E DINÂMICA",
    "SÉRIES TEMPORAIS",
    "TEORIA DOS GRAFOS",
    "TEORIA DOS JOGOS",
    "ENGENHARIA DO PRODUTO",
    "ERGONOMIA",
    "METODOLOGIA DE PROJETO DO PRODUTO",
    "PROCESSOS DE TRABALHO",
    "GERÊNCIA DO PROJETO E DO PRODUTO",
    "DESENVOLVIMENTO DE PRODUTO",
    "ENGENHARIA ECONÔMICA",
    "ESTUDO DE MERCADO",
    "LOCALIZAÇÃO INDUSTRIAL",
    "ANÁLISE DE CUSTOS",
    "ECONOMIA DE TECNOLOGIA",
    "VIDA ECONÔMICA DOS EQUIPAMENTOS",
    "AVALIAÇÃO DE PROJETOS"
]

def classificar_area(areas):
    classificacoes = {}
    for area in areas:
        pergunta = f"Qual classe a área '{area}' se enquadra?"
        resposta = obter_resposta(pergunta)
        classificacoes[area] = resposta
    return classificacoes
    
resultado = classificar_area(areas)
for area, classificacao in resultado.items():
    print(f"Área: {area}\nClassificação: {classificacao}\n")





