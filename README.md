# felicidade-idade-pais

# 📊 Análise do Bem-Estar Mundial: PIB, Demografia e Felicidade

Este repositório contém um notebook de Ciência de Dados focado no estudo das relações entre **riqueza nacional (PIB per capita)**, **demografia (idade)** e **bem-estar subjetivo (índice de felicidade / `ladder_mean`)**, combinando modelos de Regressão Linear, Regressão Logística e técnicas de validação cruzada robustas.

---

## 📌 Objetivos do Projeto

1. **Modelagem de Regressão (Parte 3):** Investigar se a relação entre idade e felicidade segue um padrão linear ou em "U" e quantificar o peso da riqueza nacional na explicação do bem-estar.
2. **Modelagem de Classificação (Parte 4):** Prever a probabilidade de um grupo populacional ser considerado "feliz" via Regressão Logística.
3. **Avaliação Sem Vazamento de Dados (*Data Leakage*):** Demonstrar o impacto de técnicas de validação adequadas (`GroupShuffleSplit` por país) em comparação com divisões aleatórias simples.
4. **Análise Crítica:** Discutir limitações de modelos estatísticos em dados agregados (falácia ecológica, extrapolações de polinômios e aplicação em políticas públicas).

---

## 🛠️ Tecnologias e Bibliotecas Utilizadas

* **Linguagem:** Python 3
* **Manipulação de Dados:** `pandas`, `numpy`
* **Visualização:** `seaborn`, `matplotlib`
* **Machine Learning & Pré-processamento:** `scikit-learn` (`Pipeline`, `StandardScaler`, `LinearRegression`, `LogisticRegression`, `RandomForestClassifier`, `GroupShuffleSplit`)
* **Padronização Geográfica:** `country_converter` (`coco`)

---

## 📐 Estrutura das Tarefas e Metodologia

### 1. Regressão Linear e Curvatura do Bem-Estar (Parte 3)
Ajuste de modelos aninhados para prever a nota de felicidade (`ladder_mean`):
* **M1:** $\hat{y} = \beta_0 + \beta_1 \cdot \text{idade}$
* **M2:** $\hat{y} = \beta_0 + \beta_1 \cdot \text{idade} + \beta_2 \cdot \text{idade}^2$
* **M3:** $\hat{y} = \text{M2} + \beta_3 \ln(\text{PIB pc}) + \beta_4 \ln(\text{pop}) + \beta_5 \ln(\text{área})$

#### Principais Achados:
* **Efeito da Idade Isolada:** A idade sozinha tem poder preditivo quase nulo sobre a nota bruta de felicidade ($R^2 \approx 1,9\%$), pois as disparidades econômicas entre os países abafam o efeito geracional.
* **Curva em "U" Local:** Ao isolar o efeito do país (`ladder_dev`), a curvatura em U do bem-estar fica evidente ($R^2 = 37,3\%$). O ponto mínimo estimado no modelo M3 ocorre por volta dos **66 anos** ($-\beta_1 / 2\beta_2$).
* **Dominância do PIB:** Com a inclusão dos controles macroeconômicos (M3), o $R^2$ salta para **62,8%** (RMSE = 0,76), confirmando que a riqueza nacional ($\beta_3 \approx +0,87$) é o fator determinante para o nível base de felicidade.
* **Outliers e Resíduos:** Países como **Afeganistão, Líbano e Botsuana** apresentam os maiores resíduos, evidenciando que o modelo não captura crises políticas, conflitos armados e colapsos institucionais.

---

### 2. Regressão Logística e Validação Agrupada (Parte 4)

#### Definição do Rótulo e Treino:
* A variável alvo `happy` foi criada utilizando a **mediana de `ladder_mean` (5,64)** como limiar, garantindo classes perfeitamente balanceadas (50% / 50%).
* O modelo foi avaliado com `GroupShuffleSplit` (agrupado por `iso3`) para testar a generalização em **países não vistos** no treino.

#### Resultados de Desempenho:
* **Acurácia no Teste:** $\approx 82,14\%$
* **AUC-ROC no Teste:** $\approx 0,9204$

#### Divisão Aleatória vs. Divisão por País (*Data Leakage*):
A comparação entre `train_test_split` aleatório e `GroupShuffleSplit` revelou que a divisão aleatória superestima drasticamente a performance (especialmente em modelos como `RandomForestClassifier`), pois o modelo memoriza a combinação fixa de PIB/população/área do país presente tanto no treino quanto no teste.

---

## 🔍 Função `check_happiness(country, age)`

A função desenvolvida permite consultar a probabilidade de felicidade para qualquer país e idade:

```python
# Exemplo de uso:
result = check_happiness(country="Brazil", age=25)
print(result)
# Saída esperada: {'country': 'Brazil', 'age': 25, 'p_happy': 0.771, 'predicted': 'happy'}