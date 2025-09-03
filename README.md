# LH_CD_Rafael_Lima
Desafio Cientista de Dados - Previsão de Notas do IMDB

# Projeto de 

Pois bem, o projeto que você está prestes a ver usa o poderoso modelo RandomForestRegressor para prever as cobiçadas notas do IMDB e, quem sabe, ajudar um estúdio de Hollywood a não gastar milhões em um filme que ninguém vai assistir.

## Bibliotecas Utilizadas

Este projeto faz uso das seguintes bibliotecas:

- **scikit-learn**: O rei dos modelos de aprendizado de máquina. O modelo de Regressão Linear, no caso, é só mais um de seus muitos talentos.
- **pandas**: Para mexer com dados sem fazer bagunças.
- **numpy**: O melhor amigo das operações numéricas. .
- **matplotlib** e **seaborn**: belezuras que são responsáveis pelas visualizações.
- **pickle**: para salvar e carregar o modelo treinado.
- **nltk & spacy**: A tropa de elite do NLP.
- 
## Pré-requisitos - Antes de Colocar a Mão na Massa

Antes de começar, você precisa garantir que tem as ferramentas certas para trabalhar, claro.  Então, instale as dependências utilizando o arquivo requirements.txt ou faça tudo de maneira manual. 

### Usando o arquivo `requirements.txt`

1. Clone este repositório ou baixe o projeto.
2. Navegue até a pasta onde o projeto está localizado.
3. Agora, basta rodar este comando mágico:
   
```bash
pip install -r requirements.txt
ou 
pip install numpy pandas scikit-learn matplotlib seaborn
```
### Estrutura do Projeto - Olha só como tudo está organizado!

```bash
.
├── data/
│   └── teste_indicium_precificacao.csv 
│
├── models/
│   └── modelo.pkl                     # O cérebro da operação!
│
├── notebooks/
│   └── LH_CD_RafaelBarbosaLima (4).ipynb # Todo o processo de EDA e modelagem.
│
├── src/
│   └── lh_cd_rafaelbarbosalima (3).py   #
│
├── README.md                          # Você está aqui! 👉 O manual de instruções.
└── requirements.txt                   # A lista de compras do projeto para o pip não reclamar.
```
