# Analise do Experimento Rapido Multiclasse

## Contexto

Este documento interpreta a execucao:

```bash
python3 -m ml.pipeline --no-tuning --no-eda --sample-size 12000 --run-id experimento_rapido
```

Objetivo do modelo:

- `Normal`
- `Flooding` = `DoS + DDoS`
- `Intrusao` = `Probe + BFA + Web-Attack + BOTNET + U2R`

O modelo utilizado foi um `MLPClassifier` com arquitetura baseline `(128, 64)`, ativacao `ReLU`, solver `Adam`, `early stopping` e dados de treino balanceados com `SMOTE`.

---

## Resumo Executivo

O experimento faz sentido e os resultados estao coerentes com o problema e com a teoria:

- o pipeline respeitou a regra de ouro do guia: `split` antes de qualquer transformacao;
- o treino foi feito so com o conjunto de treino;
- as transformacoes aprendidas no treino foram reutilizadas no teste;
- o balanceamento foi feito apenas no treino;
- o desempenho final foi alto e estavel;
- nao ha sinal forte de overfitting nesta execucao;
- as features mais importantes estao alinhadas com o comportamento esperado de ataques de rede baseados em fluxo.

Metricas principais no teste:

- `Accuracy`: `97.00%`
- `Balanced Accuracy`: `97.19%`
- `F1 Macro`: `96.48%`
- `F1 Weighted`: `97.04%`
- `MCC`: `0.9520`
- `ROC-AUC OVR Macro`: `0.9956`

Esses numeros sao muito bons para um baseline multiclasse e indicam que o problema esta bem separavel usando as features escolhidas.

---

## Pipeline Completo

O fluxo executado foi:

1. Carregar os tres CSVs do InSDN.
2. Concatenar tudo em um unico dataset.
3. Mapear os labels originais para tres classes:
   - `Normal`
   - `Flooding`
   - `Intrusao`
4. Selecionar apenas atributos relevantes de fluxo.
5. Fazer `train_test_split` estratificado em `70/30`.
6. Limpar os dados:
   - substituir valores invalidos por `NaN`;
   - imputar usando mediana aprendida no treino.
7. Aplicar `VarianceThreshold`.
8. Calcular importancia por SHAP e ranquear as features.
9. Escalonar com `StandardScaler`.
10. Balancear o treino com `SMOTE`.
11. Treinar o MLP.
12. Avaliar:
   - validacao cruzada no treino;
   - treino;
   - teste.
13. Gerar artefatos:
   - modelo salvo;
   - curvas de treino;
   - matriz de confusao;
   - relatorio de generalizacao;
   - historico de metricas.

### Por que isso esta correto segundo o guia?

- O `split` veio antes da imputacao, escalonamento e SMOTE.
- O `SMOTE` foi aplicado apenas no treino, como manda a boa pratica.
- O teste foi mantido como amostra do "mundo real".
- O overfitting foi monitorado comparando treino, validacao cruzada e teste.

---

## Leitura Etapa por Etapa do Log

## 1. Configuracoes

```text
RANDOM_STATE : 42
TEST_SIZE    : 0.3
SAMPLE_SIZE  : 12000
```

Interpretacao:

- `RANDOM_STATE=42` garante reprodutibilidade.
- `TEST_SIZE=0.30` segue a proporcao recomendada no guia.
- `sample_size=12000` significa que este foi um experimento rapido, nao o treino no dataset completo.

Isso esta correto.

---

## 2. Carregamento

Arquivos lidos:

- `Normal_data.csv`
- `OVS.csv`
- `metasploitable-2.csv`

O loader montou uma amostra estratificada de `12.000` linhas.

```text
Shape bruto consolidado : (12000, 86)
Features utilizadas     : 26
```

Interpretacao:

- As `86` colunas incluem as colunas originais mais colunas auxiliares internas do pipeline.
- O treino de fato usa `26` atributos de entrada.

Isso esta correto e esperado.

---

## 3. Split Estratificado

```text
Train: (8400, 27) | Test: (3600, 27)
Distribuicao treino: {0: 19.9, 1: 51.05, 2: 29.05}
Distribuicao teste : {0: 19.89, 1: 51.06, 2: 29.06}
```

Interpretacao:

- a proporcao das classes foi preservada quase exatamente entre treino e teste;
- isso evita distorcoes na avaliacao;
- como o problema e desbalanceado, esse passo e essencial.

Teoricamente, isso esta alinhado com classificacao supervisionada em problemas multiclasse desbalanceados.

---

## 4. Limpeza e Preparacao

```text
Valores negativos invalidos -> NaN: 193
NaN antes da imputacao: 193
Duplicatas removidas no treino: 0
NaN restantes no treino: 0
```

No teste:

```text
Valores negativos invalidos -> NaN: 68
NaN antes da imputacao: 68
NaN restantes apos imputacao: 0
```

Interpretacao:

- algumas features de tempo/taxa vieram negativas no CSV, o que nao faz sentido fisicamente;
- o pipeline tratou isso como ruido de extracao e converteu para `NaN`;
- a mediana aprendida no treino foi usada para imputar;
- nenhuma duplicata real foi removida nesta amostra.

Isso faz sentido. Em datasets de rede extraidos por ferramentas de fluxo, e comum haver pequenas anomalias numericas.

---

## 5. Selecao de Features e SHAP

```text
VarianceThreshold removeu 0 feature(s)
```

Interpretacao:

- nenhuma das 26 features escolhidas era constante nesta amostra;
- logo, nenhuma foi descartada por variancia zero.

### Ranking de importancia

Top 10:

1. `Protocol`
2. `Pkt Len Var`
3. `SYN Flag Cnt`
4. `Pkt Len Mean`
5. `Bwd IAT Min`
6. `Pkt Len Std`
7. `Flow Duration`
8. `Flow IAT Max`
9. `Flow IAT Mean`
10. `Flow IAT Min`

Interpretacao teorica:

- `Protocol` no topo faz muito sentido porque trafegos de flooding costumam ter assinatura forte em TCP/UDP/ICMP;
- variancia e media de tamanho de pacote (`Pkt Len Var`, `Pkt Len Mean`, `Pkt Len Std`) sao muito discriminativas porque ataques automatizados tendem a gerar pacotes mais padronizados;
- `SYN Flag Cnt` e muito coerente com ataques de flooding, especialmente SYN flood;
- duracao do fluxo e tempos entre pacotes (`Flow Duration`, `Flow IAT *`, `Bwd IAT *`) refletem ritmo, burst e regularidade do ataque.

### Importante

As 26 features foram mantidas. O SHAP aqui foi usado para:

- ranquear importancia;
- verificar se a intuicao de dominio fazia sentido;
- apoiar interpretabilidade.

Isso esta correto. Nao havia obrigacao de descartar features so porque foi calculado SHAP.

---

## 6. Escalonamento

```text
Media das medias (deve ser ≈0): -0.0000
Media dos desvios (deve ser ≈1): 1.0000
```

Interpretacao:

- o `StandardScaler` foi aplicado corretamente;
- isso e importante porque `MLP` e sensivel a escala dos atributos.

Teoria:

- redes neurais treinam melhor quando os atributos estao em escalas comparaveis;
- sem isso, alguns gradientes dominam os outros e o treino fica pior.

---

## 7. Balanceamento com SMOTE

Antes:

- `Normal`: `19.90%`
- `Flooding`: `51.05%`
- `Intrusao`: `29.05%`

Depois:

- `33.33%` para cada classe

Interpretacao:

- o treino original estava desbalanceado, com `Flooding` dominante;
- o `SMOTE` criou amostras sinteticas para as classes menores;
- isso ajuda o modelo a nao ficar enviesado para a classe majoritaria.

Teoria:

- em classificacao desbalanceada, modelos podem "jogar seguro" prevendo a classe dominante;
- o `SMOTE` reduz esse problema no treino.

Isso esta correto e aderente ao guia.

---

## 8. Treinamento e Validacao Cruzada

### Treinamento

```text
Arquitetura : (128, 64)
Solver      : adam
Ativacao    : relu
Alpha       : 0.001
max_iter    : 250
Treinamento concluido em 36.6s
Epocas executadas : 70
Loss final        : 0.070915
```

Interpretacao:

- o modelo nao precisou das 250 epocas maximas;
- ele parou em `70`, o que indica que o `early stopping` funcionou;
- a `loss final` ficou baixa, o que e bom.

Isso sugere convergencia saudavel.

### Validacao cruzada

```text
accuracy          : 0.9611 +/- 0.0101
balanced_accuracy : 0.9610 +/- 0.0101
f1_macro          : 0.9610 +/- 0.0102
precision_macro   : 0.9619 +/- 0.0097
recall_macro      : 0.9610 +/- 0.0101
```

Interpretacao:

- o desempenho medio em `3-fold CV` foi muito bom;
- o desvio padrao em torno de `1%` e baixo, sinal de estabilidade;
- `F1 Macro ~ 0.961` indica que o modelo generaliza bem no treino, sem depender de uma unica classe.

Isso e muito mais informativo do que olhar apenas a acuracia.

---

## 9. Avaliacao em Treino e Teste

## Treino

- `Accuracy`: `97.29%`
- `Balanced Accuracy`: `97.64%`
- `F1 Macro`: `96.83%`
- `MCC`: `0.9568`
- `ROC-AUC`: `0.9985`

## Teste

- `Accuracy`: `97.00%`
- `Balanced Accuracy`: `97.19%`
- `F1 Macro`: `96.48%`
- `MCC`: `0.9520`
- `ROC-AUC`: `0.9956`

### O que isso significa?

#### Accuracy

Percentual total de classificacoes corretas.

Aqui ficou muito alta (`97%`), mas sozinha nao basta porque as classes nao sao perfeitamente balanceadas.

#### Balanced Accuracy

Media do recall por classe.

Essa metrica e mais robusta em problemas desbalanceados. Como ficou `97.19%`, significa que o modelo foi bem nas tres classes, e nao apenas na majoritaria.

#### F1 Macro

Media simples do F1 entre as classes.

Ela vale muito aqui porque trata `Normal`, `Flooding` e `Intrusao` com o mesmo peso. O valor de `96.48%` e excelente.

#### F1 Weighted

Media ponderada pelo tamanho das classes.

Como ficou muito proxima do `F1 Macro`, isso e um bom sinal: o desempenho nao dependeu so da classe maior.

#### MCC

Matthews Correlation Coefficient mede qualidade global da classificacao e e bem rigoroso.

`0.9520` e muito alto. Quanto mais perto de `1`, melhor.

#### ROC-AUC OVR Macro

Mede capacidade do modelo de separar as classes em um esquema "one-vs-rest".

`0.9956` indica separabilidade muito forte entre as classes.

---

## Existe overfitting?

Pelo relatorio de generalizacao:

- `accuracy_gap`: `0.00286`
- `balanced_accuracy_gap`: `0.00447`
- `f1_macro_gap`: `0.00341`
- `mcc_gap`: `0.00476`
- `roc_auc_gap`: `0.00295`

Interpretacao:

- os gaps entre treino e teste sao muito pequenos;
- isso e exatamente o que queremos ver;
- portanto, **nao ha evidencia forte de overfitting** neste experimento.

### Um detalhe importante

O treino e um pouco melhor que o teste, como esperado.

Mas a diferenca e tao pequena que o modelo parece estar generalizando bem, e nao apenas memorizando o treino.

---

## Interpretacao do Relatorio por Classe

## Classe `Normal`

No teste:

- `precision = 0.8977`
- `recall = 0.9804`
- `f1 = 0.9372`

Interpretacao:

- quase todos os fluxos normais reais foram reconhecidos como normais;
- porem, quando o modelo previu `Normal`, em alguns casos ele estava errado.

Isso acontece porque alguns ataques foram confundidos com `Normal`.

Do ponto de vista de seguranca, esta e a classe mais sensivel, porque prever `Normal` incorretamente pode significar deixar um ataque passar.

## Classe `Flooding`

No teste:

- `precision = 0.9905`
- `recall = 0.9668`
- `f1 = 0.9785`

Interpretacao:

- excelente desempenho;
- faz sentido, pois flooding costuma ter padrao muito forte em taxa, flags e tempos entre pacotes.

## Classe `Intrusao`

No teste:

- `precision = 0.9893`
- `recall = 0.9685`
- `f1 = 0.9787`

Interpretacao:

- tambem excelente;
- isso mostra que o grupo `Probe + BFA + Web-Attack + BOTNET + U2R`, mesmo sendo heterogeneo, ainda gera um padrao estatistico aprendivel no espaco de features escolhido.

---

## Matriz de Confusao do Teste

Matriz:

```text
                Predito
Real          Normal  Flooding  Intrusao
Normal          702       8         6
Flooding         56    1777         5
Intrusao         24       9      1013
```

### Leitura por linha

#### Normal

- `702/716` normais foram classificados corretamente;
- `14` foram classificados como ataque.

Isso significa:

- baixa taxa de falso alarme sobre trafego legitimo;
- aproximadamente `1.95%` dos normais foram sinalizados como ataque.

#### Flooding

- `1777/1838` foram reconhecidos corretamente;
- `56` foram confundidos com `Normal`;
- `5` foram confundidos com `Intrusao`.

Do ponto de vista operacional:

- o pior erro aqui e `Flooding -> Normal`, porque e ataque perdido;
- isso aconteceu em `56` casos, cerca de `3.05%` da classe.

#### Intrusao

- `1013/1046` foram reconhecidos corretamente;
- `24` foram confundidos com `Normal`;
- `9` foram confundidos com `Flooding`.

Tambem aqui o erro mais sensivel e `Intrusao -> Normal`.

### Ataques que passaram como benignos

Somando os ataques classificados como `Normal`:

- `Flooding -> Normal = 56`
- `Intrusao -> Normal = 24`

Total:

- `80` ataques classificados como benignos no teste

Como o total de ataques no teste e `1838 + 1046 = 2884`, isso significa:

- cerca de `2.77%` dos ataques passaram como benignos

Isso nao e zero, mas e um resultado bom para um baseline multiclasse.

---

## Leitura dos Graficos Gerados

## 1. `feature_importance_multiclass.png`

Mostra as features mais importantes.

Interpretacao:

- `Protocol` domina o ranking;
- tamanho de pacote, variabilidade e flags TCP aparecem logo em seguida;
- tempos entre pacotes e duracao do fluxo tambem aparecem fortes.

Isso esta muito alinhado com a teoria de deteccao por fluxo:

- flooding altera taxas, duracao, burst e flags;
- intrusao altera padroes de interacao, tamanho e periodicidade.

## 2. `loss_curve_baseline.png`

O grafico mostra:

- `loss de treino` caindo continuamente;
- `score de validacao interna` subindo e estabilizando em patamar alto;
- parada em `70` epocas.

Interpretacao:

- o modelo convergiu;
- nao ha sinal visual de divergencia;
- o `early stopping` parece ter evitado treino excessivo.

## 3. `learning_curve_baseline.png`

O grafico mostra:

- curva de treino e validacao proximas;
- ambas melhoram com mais dados;
- o gap entre elas e pequeno.

Interpretacao:

- bom sinal de generalizacao;
- nao parece haver overfitting forte;
- ainda existe ganho com mais dados, o que e esperado.

## 4. `generalization_gap_baseline.png`

Mostra treino e teste lado a lado para:

- Accuracy
- Balanced Accuracy
- F1 Macro
- F1 Weighted
- MCC
- ROC-AUC

Interpretacao:

- as barras de treino e teste sao muito proximas;
- isso confirma numericamente e visualmente que o modelo generaliza bem.

---

## Faz sentido que o teste tenha ido um pouco melhor que a validacao cruzada?

Sim.

Aqui temos:

- `CV F1 Macro`: `0.9610`
- `Teste F1 Macro`: `0.9648`

Essa diferenca e pequena.

Isso pode acontecer por:

- variacao estatistica normal;
- esse split especifico de teste ter ficado ligeiramente mais favoravel;
- `SMOTE` e o comportamento do MLP variarem um pouco entre dobras e treino final.

O importante e que:

- nao ha discrepancia grande;
- os valores estao coerentes entre si.

---

## Paralelo com a Teoria

## Por que essas features funcionam?

Ataques de rede mudam a estrutura estatistica do fluxo.

### Flooding

Tende a produzir:

- alta taxa de pacotes;
- alta regularidade temporal;
- muitos pacotes semelhantes;
- padroes fortes de protocolo e flags.

Por isso `Protocol`, `SYN Flag Cnt`, `Flow Duration`, `Flow IAT *`, `Bwd Pkts/s` e `Flow Pkts/s` aparecem relevantes.

### Intrusao

Mesmo sendo mais heterogenea, costuma afetar:

- sequencia temporal das trocas;
- direcionalidade do trafego;
- volume e tamanho dos pacotes;
- assimetria entre ida e volta.

Por isso `Pkt Len Mean`, `Pkt Len Var`, `Bwd IAT *`, `Down/Up Ratio` e totais de bytes/pacotes ajudam bastante.

---

## Significado de Cada Atributo Usado no Treinamento

## 1. `Protocol`

Identificador do protocolo da camada de transporte/rede, por exemplo TCP, UDP ou ICMP.

## 2. `Pkt Len Var`

Variancia do tamanho dos pacotes do fluxo. Mede o quanto os tamanhos variam.

## 3. `SYN Flag Cnt`

Quantidade de pacotes com flag TCP `SYN`. Muito util para detectar tentativas de abertura massiva de conexao.

## 4. `Pkt Len Mean`

Media do tamanho dos pacotes do fluxo.

## 5. `Bwd IAT Min`

Menor tempo entre chegadas de pacotes no sentido backward, isto e, do destino para a origem.

## 6. `Pkt Len Std`

Desvio padrao do tamanho dos pacotes. Complementa media e variancia.

## 7. `Flow Duration`

Duracao total do fluxo.

## 8. `Flow IAT Max`

Maior tempo entre chegadas de pacotes considerando o fluxo inteiro.

## 9. `Flow IAT Mean`

Media dos tempos entre chegadas de pacotes no fluxo.

## 10. `Flow IAT Min`

Menor tempo entre chegadas de pacotes no fluxo.

## 11. `TotLen Bwd Pkts`

Quantidade total de bytes no sentido backward.

## 12. `Tot Bwd Pkts`

Quantidade total de pacotes no sentido backward.

## 13. `Bwd IAT Tot`

Soma total dos intervalos de tempo entre pacotes backward.

## 14. `Tot Fwd Pkts`

Quantidade total de pacotes no sentido forward.

## 15. `TotLen Fwd Pkts`

Quantidade total de bytes no sentido forward.

## 16. `Bwd Pkts/s`

Taxa de pacotes por segundo no sentido backward.

## 17. `Flow Pkts/s`

Taxa total de pacotes por segundo do fluxo.

## 18. `Flow IAT Std`

Desvio padrao dos tempos entre chegadas do fluxo.

## 19. `Bwd IAT Max`

Maior intervalo entre chegadas de pacotes no sentido backward.

## 20. `Down/Up Ratio`

Razao entre o trafego de retorno e o trafego de ida. Ajuda a captar assimetria no fluxo.

## 21. `Flow Byts/s`

Taxa total de bytes por segundo do fluxo.

## 22. `Bwd IAT Std`

Desvio padrao dos tempos entre chegadas backward.

## 23. `ACK Flag Cnt`

Quantidade de pacotes TCP com flag `ACK`.

## 24. `Bwd IAT Mean`

Media do intervalo entre pacotes no sentido backward.

## 25. `Idle Mean`

Media dos periodos ociosos do fluxo.

## 26. `Active Mean`

Media dos periodos ativos do fluxo.

---

## O que esta mais forte e o que merece atencao

## Pontos fortes

- pipeline correto do ponto de vista metodologico;
- metrica multiclasse muito boa;
- baixo gap treino/teste;
- curvas visuais saudaveis;
- pouca dependencia de uma unica classe;
- feature importance coerente com teoria de trafego de rede.

## Ponto de atencao principal

A classe `Normal` tem a menor `precision`.

Isso significa:

- quando o modelo prediz `Normal`, ele ainda erra mais do que nas outras classes;
- em seguranca, isso importa porque pode significar ataque sendo tratado como benigno.

Mesmo assim, o volume desse erro ficou relativamente baixo para baseline.

---

## Conclusao

O experimento `experimento_rapido` esta correto, consistente e metodologicamente bem implementado.

Os resultados fazem sentido:

- o problema esta bem separavel no espaco de features escolhido;
- o modelo aprendeu bem os padroes de `Flooding` e `Intrusao`;
- o comportamento entre treino, validacao e teste sugere boa generalizacao;
- as features mais importantes estao alinhadas com o que a teoria de analise de trafego por fluxo preve.

Em resumo:

- **sim, os resultados fazem sentido**
- **sim, o pipeline esta coerente com o guia**
- **sim, a execucao parece correta**

O proximo passo natural seria um destes:

1. rodar no dataset completo;
2. executar tuning;
3. comparar outras arquiteturas baselines;
4. aprofundar a analise dos erros da classe `Normal`.
