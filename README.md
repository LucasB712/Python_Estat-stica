# 📊 Estatísticas em Python: Métricas e Análises 🧮

Este repositório tem como objetivo aprender e aplicar as principais métricas estatísticas e técnicas de análise de dados utilizando Python e bibliotecas populares como NumPy, Pandas, Matplotlib, Seaborn, e SciPy. O conteúdo é voltado tanto para iniciantes quanto para aqueles que desejam aprofundar seus conhecimentos em estatísticas aplicadas.

## 🎯 Objetivo do Projeto

Explorar e calcular métricas estatísticas como média, mediana, moda, desvio padrão, etc.

Realizar análises exploratórias de dados (EDA) utilizando visualizações gráficas.

Aplicar testes de hipóteses e outros conceitos estatísticos para análise de dados reais.

Compreender distribuições de dados e como elas influenciam a modelagem estatística.

##  🚀 Tecnologias Usadas

Python 3 (linguagem de programação principal)

NumPy (para operações numéricas)

Pandas (para manipulação e análise de dados)

Matplotlib (para visualização de gráficos)

Seaborn (para visualizações estatísticas)

SciPy (para testes estatísticos e distribuições)

Jupyter Notebook (para interatividade e visualização)

## 📋 Instalação
1. Instalar Dependências

Para instalar todas as dependências necessárias, basta rodar o seguinte comando:

```bash
pip install numpy pandas matplotlib seaborn scipy jupyter
```

2. Rodando o Jupyter Notebook

Este repositório contém Jupyter Notebooks com exemplos interativos. Para iniciar o Jupyter Notebook, execute:

```bash 
jupyter notebook
```


## 📚 Conteúdo das Aulas/Notebooks
1. Cálculo de Métricas Estatísticas Básicas

No notebook 01_metrica_basica.ipynb, abordamos as principais métricas utilizadas na estatística descritiva, como:

Média (Mean)

Mediana (Median)

Moda (Mode)

Desvio padrão (Standard Deviation)

Variância (Variance)

Quartis e Outliers (Q1, Q3, IQR)

Exemplo de código:

import numpy as np
import pandas as pd

## Exemplo de dados
data = [12, 15, 13, 10, 10, 16, 15, 14, 13, 18]

## Cálculo da média
mean = np.mean(data)
print(f'Média: {mean}')

## Cálculo do desvio padrão
std_dev = np.std(data)
print(f'Desvio padrão: {std_dev}')

2. Análise Exploratória de Dados (EDA)

O notebook 02_analise_exploratoria.ipynb ensina a explorar e visualizar os dados usando gráficos e estatísticas descritivas para entender padrões e características dos dados. Utilizamos Pandas, Matplotlib e Seaborn.

Principais tópicos abordados:

Histograma: Distribuição de frequências

Boxplot: Visualização de quartis e outliers

Gráfico de dispersão: Relação entre duas variáveis

Correlação: Estudo da relação entre variáveis numéricas

Exemplo de código:

import seaborn as sns
import matplotlib.pyplot as plt

### Carregar um dataset de exemplo do Seaborn
data = sns.load_dataset('tips')

### Gráfico de dispersão
sns.scatterplot(x='total_bill', y='tip', data=data)
plt.show()

### Boxplot para visualização de outliers
sns.boxplot(x='day', y='total_bill', data=data)
plt.show()

3. Testes de Hipóteses

No notebook 03_testes_hipotese.ipynb, introduzimos os principais testes de hipóteses utilizados em estatísticas inferenciais, como:

Teste t de Student

Teste Qui-quadrado

Teste de Mann-Whitney

Exemplo de código (Teste t de Student):

from scipy import stats

### Exemplo de duas amostras
amostra1 = [2.5, 3.0, 2.8, 3.2, 2.7]
amostra2 = [3.5, 3.6, 3.3, 3.7, 3.2]

### Teste t de Student para amostras independentes
t_stat, p_value = stats.ttest_ind(amostra1, amostra2)

print(f'Estatística t: {t_stat}')
print(f'Valor p: {p_value}')

4. Regressão Linear Simples

No notebook 04_regressao_linear.ipynb, aprendemos sobre regressão linear simples, que é uma técnica para modelar a relação entre duas variáveis.

Exemplo de código:

import seaborn as sns
import statsmodels.api as sm

### Carregar dataset de exemplo
data = sns.load_dataset('tips')

### Definindo a variável dependente e independente
X = data['total_bill']
y = data['tip']

### Adicionando uma constante (intercepto)
X = sm.add_constant(X)

### Modelo de regressão linear
modelo = sm.OLS(y, X).fit()

### Resultados do modelo
print(modelo.summary())

## 🛠 Como Contribuir

Se você deseja contribuir para o projeto, siga os seguintes passos:

Faça um fork do repositório.

Crie uma branch para suas alterações:

```bash
git checkout -b minha-nova-funcionalidade
```

Faça commit das suas alterações:

```bash
git commit -m "Descrição das alterações"
```

Push para o seu repositório:

```bash
git push origin minha-nova-funcionalidade
```

Abra um pull request para o repositório principal.

# 📄 Licença

Este projeto está licenciado sob a MIT License
.
