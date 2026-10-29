# Detecção de Fraude em Cartão de Crédito

Projeto final do curso de detecção de fraudes da DIO, conduzido pela Expert Dra. Isadora Garcia Ferrão. O notebook treina, compara e explica modelos de classificação para identificar transações fraudulentas em uma base real de cartão de crédito.

## 1. O problema

A base tem **284.807 transações**, das quais apenas **492 são fraude (0,173%)**. Para cada fraude existem cerca de 577 transações normais.

Nesse cenário, a **acurácia engana**: um modelo que responde "não é fraude" para tudo acerta 99,83% das vezes e não detecta nenhuma fraude. Os resultados deste projeto mostram isso na prática. A Regressão Logística teve **97,3% de acurácia**, mas só **5,5% de precisão** na classe fraude: de cada 100 transações que ela bloqueou, apenas 5 eram fraude de verdade.

Por isso, a avaliação foca **somente na classe fraude**:

| Métrica | Pergunta que responde |
| --- | --- |
| **Recall** (métrica principal) | De todas as fraudes que existiam, quantas o modelo pegou? |
| **Precisão** | De tudo que o modelo marcou como fraude, quanto era fraude de verdade? |
| **F1** | Qual o equilíbrio entre recall e precisão? |
| **PR AUC** | Qual a qualidade do modelo em todos os limiares possíveis? |

## 2. Dados

- **Fonte:** [creditcard.csv](https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv), carregado diretamente pelo link com `pandas.read_csv`. O arquivo não está neste repositório.
- **Origem:** transações de portadores de cartão europeus em dois dias de setembro de 2013.
- **Colunas:** `Time` (segundos desde a primeira transação), `Amount` (valor), `V1` a `V28` (anonimizadas por PCA) e `Class` (0 = normal, 1 = fraude).
- **Qualidade:** nenhum valor nulo.

## 3. Preparação dos dados

1. **`log_amount = log(1 + Amount)`**: os valores são muito assimétricos (a maioria é pequena, mas há transações acima de 25 mil). O log comprime essa cauda. O `+1` evita erro em transações de valor zero. A coluna `Amount` original foi removida para não duplicar a informação.
2. **`hora = (Time // 3600) % 24`**: hora do dia **relativa**, de 0 a 23. A base não informa o horário real de início, então "hora 0" não é necessariamente meia-noite.
3. **Separação treino/teste (80/20) com `stratify=y`**, mantendo a mesma proporção de fraudes nos dois conjuntos:

    | Conjunto | Transações | Fraudes | % fraude |
    | --- | --- | --- | --- |
    | Treino | 227.845 | 394 | 0,173% |
    | Teste | 56.962 | 98 | 0,172% |

4. **Padronização com `StandardScaler`**, ajustado **somente no treino** e aplicado ao teste, para evitar vazamento de informação.

## 4. Modelos

Os três modelos receberam peso maior para a classe fraude, de forma que errar uma fraude "custe" muito mais do que errar uma transação normal:

- **Regressão Logística** (baseline): `class_weight="balanced"`
- **Random Forest**: 100 árvores, `class_weight="balanced_subsample"`
- **XGBoost**: 300 árvores, profundidade 4, `scale_pos_weight=577`

## 5. Resultados (limiar padrão 0,5)

Métricas da **classe fraude** no conjunto de teste:

| Modelo | Recall | Precisão | F1 | ROC AUC | PR AUC |
| --- | --- | --- | --- | --- | --- |
| Regressão Logística | 0,908 | 0,055 | 0,104 | 0,971 | 0,713 |
| Random Forest | 0,776 | **0,938** | **0,849** | 0,963 | **0,875** |
| XGBoost | 0,857 | 0,824 | 0,840 | **0,985** | 0,871 |

**Leitura:**

- A Regressão Logística tem o maior recall, mas a precisão de 5,5% a torna inviável: ela bloquearia muitos clientes honestos.
- Random Forest e XGBoost ficaram praticamente empatados em PR AUC (0,875 × 0,871). O Random Forest é mais conservador (mais precisão, menos recall) e o XGBoost é mais equilibrado.
- **A curva ROC engana em dados desbalanceados.** Pela ROC AUC, a Regressão Logística (0,971) parece melhor que o Random Forest (0,963). A curva precisão-recall mostra o contrário com clareza: 0,713 contra 0,875. Por isso a PR AUC foi usada para escolher o modelo.

## 6. Limiar de decisão

Por padrão, o modelo marca fraude quando a probabilidade passa de 0,5. Como o recall é a métrica principal, ajustei o limiar do **Random Forest** (melhor PR AUC).

**Critério:** o limiar que garante **recall de pelo menos 85% com a maior precisão possível**.

**Método:** o limiar foi escolhido com **validação cruzada estratificada (5 dobras) somente no treino**. O conjunto de teste não foi usado nessa escolha, então o resultado abaixo é uma estimativa honesta.

**Limiar escolhido: 0,10**

| Limiar | Recall | Precisão | F1 |
| --- | --- | --- | --- |
| 0,50 (padrão) | 0,776 | 0,938 | 0,849 |
| **0,10 (escolhido)** | **0,888** | 0,690 | 0,777 |

Em casos concretos, nas 98 fraudes e 56.864 transações normais do teste:

| Limiar | Fraudes detectadas | Fraudes que passaram | Alarmes falsos |
| --- | --- | --- | --- |
| 0,50 | 76 | 22 | 5 |
| **0,10** | **87** | **11** | 39 |

Baixar o limiar **reduziu pela metade as fraudes não detectadas** (de 22 para 11), ao custo de 34 alarmes falsos a mais, cerca de 3 alarmes por fraude extra detectada. Os 39 alarmes falsos representam 0,07% das transações normais. Considerando que uma fraude não detectada costuma custar muito mais do que verificar uma transação legítima, a troca compensa.

## 7. Explicabilidade

### Importância das variáveis

Random Forest e XGBoost concordam na variável mais importante: **V14**, seguida de **V10** e **V4**. No XGBoost, V14 concentra mais da metade da importância total.

### SHAP

O SHAP foi aplicado ao XGBoost, que tem desempenho equivalente ao Random Forest e uma implementação de SHAP muito mais rápida.

**Visão geral (beeswarm, 2.000 transações do teste):**

- **V14 baixo** é o sinal mais forte de fraude.
- **V4 alto** empurra a decisão para fraude.
- **V12 e V10 baixos** também indicam fraude.
- **V11 alto** contribui para fraude.
- `log_amount` aparece entre as 10 mais relevantes, mas com efeito pequeno. O valor da transação, sozinho, diz pouco.

**Uma transação individual (waterfall):** para uma fraude detectada pelo modelo, a decisão foi puxada principalmente por V14 (+4,44), V10 (+2,06), V12 (+1,82) e V17 (+1,65), todas com valores muito baixos. A saída do modelo foi de 9,70 em log-odds, partindo de uma média de -0,05, o que corresponde a uma probabilidade de fraude próxima de 100%.

Como V1 a V28 foram anonimizadas por PCA, não é possível dizer o que cada uma representa no mundo real. O SHAP ainda assim mostra **quais sinais** o modelo usa e **em que direção**, o que permite auditar as decisões.

## 8. Balanceamento: peso de classe × undersampling × SMOTE

Comparei três formas de lidar com o desbalanceamento, usando a Regressão Logística para isolar o efeito da técnica. A reamostragem foi aplicada **somente no treino**. O teste manteve a proporção real.

| Estratégia | Treino (normais / fraudes) | Recall | Precisão | F1 | PR AUC |
| --- | --- | --- | --- | --- | --- |
| Peso de classe | 227.451 / 394 | 0,908 | 0,055 | 0,104 | 0,713 |
| Undersampling | 394 / 394 | 0,918 | 0,037 | 0,071 | 0,610 |
| SMOTE (oversampling) | 227.451 / 227.451 | 0,918 | 0,054 | 0,102 | **0,719** |

**Conclusão:** o recall ficou praticamente igual nas três. O **undersampling** foi o pior, porque descartar 99,8% das transações normais faz o modelo perder informação, e a PR AUC cai para 0,610. **SMOTE e peso de classe empataram**, mas o peso de classe é mais simples e não cria dados sintéticos. Por isso ele foi a escolha nos modelos principais.

## 9. O que mudei em relação à Expert

- **Ajuste de limiar no Random Forest**, uma das evoluções sugeridas pela Expert.
- **Limiar escolhido por validação cruzada no treino**, sem olhar o conjunto de teste, com um critério de negócio explícito (recall ≥ 85%).
- **Comparação entre peso de classe, undersampling e SMOTE**, medida pelo recall e pela PR AUC.
- **Nova variável `hora`**, derivada de `Time`.
- **Tradução das métricas em casos concretos** (fraudes detectadas, fraudes perdidas e alarmes falsos), para facilitar a decisão de negócio.

## 10. Limitações e próximos passos

- A variável `hora` é relativa e não apareceu entre as mais importantes. Variáveis de comportamento ao longo do tempo, como várias transações em sequência, podem trazer mais sinal.
- Não houve busca de hiperparâmetros. Um `GridSearchCV` focado em PR AUC pode melhorar os modelos de árvore.
- O limiar ideal depende do custo real de cada erro, que deveria ser definido junto com a área de negócio.

## Como executar

1. Abra o notebook no [Google Colab](https://colab.research.google.com/).
2. Execute **Ambiente de execução → Executar tudo**. O dataset é baixado automaticamente pelo link.

## Tecnologias

Python, pandas, NumPy, Matplotlib, scikit-learn, XGBoost, SHAP e imbalanced-learn.
