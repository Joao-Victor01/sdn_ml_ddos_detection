# Analise Detalhada dos Resultados de ML

## 1. Objetivo deste Documento

Este documento apresenta uma analise detalhada dos resultados gerados pelo pipeline de ML, com foco em:

- o significado de cada metrica;
- por que essas metricas foram escolhidas;
- o que elas dizem sobre o comportamento do modelo;
- se o modelo aprendeu padroes reais ou apenas decorou os dados;
- se ha sinais de overfitting;
- se ha sinais de data leakage;
- o que podemos concluir sobre baseline e modelo otimizado.

Os resultados analisados aqui estao em:

- `outputs/runs/baseline_full/`
- `outputs/runs/tuned_full/`
- historico consolidado em `outputs/metrics_history.json`

---

## 2. Quais Resultados Estao Sendo Analisados

## 2.1 Baseline completo

Run:

- `tuned_full` no historico, com label `MLP Baseline (Teste)`

Metricas de teste:

- `Accuracy`: `0.982029`
- `Balanced Accuracy`: `0.982494`
- `Precision Macro`: `0.974236`
- `Recall Macro`: `0.982494`
- `F1 Macro`: `0.978201`
- `F1 Weighted`: `0.982151`
- `MCC`: `0.971069`
- `G-Mean`: `0.987220`
- `ROC-AUC OVR Macro`: `0.999377`

## 2.2 Modelo otimizado

Run:

- `tuned_tuned_full` no historico, com label `MLP Otimizado (Teste)`

Metricas de teste:

- `Accuracy`: `0.981147`
- `Balanced Accuracy`: `0.978878`
- `Precision Macro`: `0.975203`
- `Recall Macro`: `0.978878`
- `F1 Macro`: `0.976945`
- `F1 Weighted`: `0.981229`
- `MCC`: `0.969462`
- `G-Mean`: `0.984795`
- `ROC-AUC OVR Macro`: `0.999191`

---

## 3. Primeira Conclusao: o Baseline Ja e Muito Forte

Antes de discutir as metricas individualmente, ha uma leitura global importante:

- o baseline completo ja alcancou desempenho excelente;
- o tuning nao trouxe ganho agregado;
- na verdade, o tuning piorou levemente quase todas as metricas globais.

Isso e relevante porque mostra que:

- o problema esta bem estruturado;
- as features escolhidas sao muito informativas;
- a arquitetura baseline ja e suficiente para capturar grande parte da separacao entre as classes;
- tuning nao e garantia de melhora.

Do ponto de vista metodologico, isso e um resultado legitimo e ate desejavel: um baseline forte e preferivel a uma melhoria artificial ou instavel.

---

## 4. Por que Usamos Essas Metricas

## 4.1 Por que nao usar apenas Accuracy?

Porque o problema e multiclasse e desbalanceado.

Distribuicao total das classes:

- `Normal`: `68424`
- `Flooding`: `175558`
- `Intrusao`: `99907`

Se olhassemos apenas `Accuracy`, um modelo poderia ir muito bem na classe dominante e ainda esconder um comportamento ruim nas classes menores.

Por isso, `Accuracy` foi mantida, mas nao como unica referencia.

---

## 4.2 Accuracy

### O que mede

Proporcao total de classificacoes corretas:

```text
acertos / total de amostras
```

### O que ela diz aqui

- Baseline: `98.20%`
- Otimizado: `98.11%`

### Interpretacao

O modelo acerta a imensa maioria das amostras.

Mas sozinha essa metrica nao responde:

- se todas as classes estao sendo tratadas de forma equilibrada;
- se o modelo esta "sacrificando" a classe `Normal` para ganhar desempenho nas outras;
- se o modelo e robusto em termos de distribuicao de erros.

### Por que mantemos essa metrica?

Porque ela e intuitiva e comunica bem o desempenho global.

---

## 4.3 Balanced Accuracy

### O que mede

A media do recall por classe.

Ela pergunta:

- "em media, quanto o modelo reconhece corretamente de cada classe?"

### Por que ela e importante

Em problemas desbalanceados, `Balanced Accuracy` e mais justa do que `Accuracy`, porque nao deixa a classe majoritaria dominar a leitura.

### Resultados

- Baseline: `98.2494%`
- Otimizado: `97.8878%`

### Interpretacao

O baseline conseguiu manter excelente sensibilidade media nas tres classes.

Como essa metrica caiu no modelo otimizado, isso significa que:

- o tuning nao melhorou a cobertura media por classe;
- alguma classe perdeu recall, mesmo que outra possa ter melhorado.

---

## 4.4 Precision Macro

### O que mede

A media da precisao entre as classes.

Precisao, por classe, responde:

- "quando o modelo previu esta classe, com que frequencia ele acertou?"

### Por que usar versao macro

Porque a versao macro da o mesmo peso para todas as classes, independentemente do tamanho.

### Resultados

- Baseline: `97.4236%`
- Otimizado: `97.5203%`

### Interpretacao

Aqui o otimizado foi ligeiramente melhor.

Isso significa que, em media:

- quando ele toma uma decisao, ele comete um pouco menos de "rotulacoes indevidas" por classe.

Mas isso nao quer dizer que ele seja melhor no geral, porque precisao so conta uma parte da historia.

---

## 4.5 Recall Macro

### O que mede

A media do recall entre as classes.

Recall por classe responde:

- "de tudo o que realmente pertencia a esta classe, quanto o modelo conseguiu recuperar?"

### Por que ele e importante aqui

Em seguranca, recall e critico porque representa:

- capacidade de nao deixar casos verdadeiros passarem despercebidos.

### Resultados

- Baseline: `98.2494%`
- Otimizado: `97.8878%`

### Interpretacao

O baseline foi melhor.

Isso sugere que o tuning introduziu um comportamento um pouco mais conservador:

- ele passou a errar menos em algumas predicoes positivas;
- mas deixou escapar mais exemplos verdadeiros de determinadas classes.

Em seguranca, esse trade-off precisa ser analisado com muito cuidado.

---

## 4.6 F1 Macro

### O que mede

O F1 e a media harmonica entre precision e recall.

A versao macro da:

- o mesmo peso a `Normal`, `Flooding` e `Intrusao`.

### Por que essa e uma das metricas centrais do projeto

Porque ela equilibra:

- "nao acusar errado demais" e
- "nao deixar passar casos verdadeiros"

sem deixar a classe majoritaria dominar a leitura.

### Resultados

- Baseline: `97.8201%`
- Otimizado: `97.6945%`

### Interpretacao

O baseline foi melhor.

Como o `F1 Macro` e a metrica central para este tipo de problema, esse resultado indica que:

- o baseline oferece melhor equilibrio geral entre classes;
- o tuning nao melhorou a qualidade operacional global.

### Por que ela e melhor que so olhar accuracy

Porque accuracy poderia continuar alta mesmo com degradacao relevante em uma classe importante.

O `F1 Macro` torna isso visivel.

---

## 4.7 F1 Weighted

### O que mede

Tambem e media entre precision e recall, mas ponderada pelo tamanho das classes.

### Por que usar junto com o F1 Macro

Para comparar:

- `F1 Macro`: equilibrio entre classes
- `F1 Weighted`: desempenho ponderado pela distribuicao real

### Resultados

- Baseline: `98.2151%`
- Otimizado: `98.1229%`

### Interpretacao

Novamente o baseline venceu.

Como `F1 Weighted` e maior do que `F1 Macro`, isso indica que:

- as classes mais frequentes estao sendo tratadas muito bem;
- mas ainda existe um pequeno custo relativo nas classes menos favorecidas.

Mesmo assim, a diferenca entre os dois F1 e pequena, o que e um bom sinal.

---

## 4.8 MCC

### O que mede

`Matthews Correlation Coefficient` mede a qualidade global da classificacao levando em conta acertos e erros de forma mais rigorosa.

Ele e valioso porque:

- nao e facilmente "enganado" por distribuicoes desbalanceadas;
- resume melhor a estrutura da matriz de confusao.

### Intervalo

- `1` = classificacao perfeita
- `0` = desempenho aleatorio
- `-1` = classificacao totalmente inversa

### Resultados

- Baseline: `0.971069`
- Otimizado: `0.969462`

### Interpretacao

Os dois modelos sao excelentes.

Mas o baseline continua superior.

Como o MCC continua muito alto, isso e forte evidencia de que o modelo:

- nao esta apenas explorando a classe majoritaria;
- esta realmente modelando bem a estrutura do problema.

---

## 4.9 G-Mean

### O que mede

O `Geometric Mean` procura refletir equilibrio entre desempenhos por classe.

Ele e util quando queremos evitar que uma classe fique muito sacrificada.

### Resultados

- Baseline: `0.987220`
- Otimizado: `0.984795`

### Interpretacao

Mais uma vez, o baseline foi melhor.

Isso reforca a leitura de que:

- o tuning deslocou o comportamento do modelo;
- esse deslocamento nao foi vantajoso de forma agregada.

---

## 4.10 ROC-AUC OVR Macro

### O que mede

Mede a capacidade de separacao entre classes em esquema `one-vs-rest`.

Em termos intuitivos:

- o quanto o modelo consegue atribuir probabilidade maior a classes corretas do que a incorretas.

### Por que usar

Porque ela avalia a qualidade da ordenacao das probabilidades, nao apenas a classe final.

### Resultados

- Baseline: `0.999377`
- Otimizado: `0.999191`

### Interpretacao

Os dois modelos separam extremamente bem as classes em termos probabilisticos.

Isso sugere que o espaco de features e muito informativo.

No entanto, uma AUC muito alta nao prova sozinha que a decisao final esta perfeita, porque:

- a regra de decisao final ainda pode cometer erros localizados;
- calibracao e limiar importam.

Mesmo assim, aqui o resultado e excelente.

---

## 4.11 Por que nao priorizamos outras metricas?

### Por que nao usar apenas micro-F1?

Porque micro-F1 tende a se aproximar muito da acuracia em problemas multiclasse e pode mascarar desbalanceamento.

### Por que nao usar somente precision ou somente recall?

Porque cada uma so mostra metade da historia.

### Por que nao usar PR-AUC como metrica principal?

Poderia ser util, especialmente em problemas muito desbalanceados, mas o conjunto atual de metricas ja cobre bem:

- desempenho global;
- equilibrio entre classes;
- qualidade da discriminacao;
- risco de viés para classe dominante.

---

## 5. O que as Matrizes de Confusao Revelam

## 5.1 Baseline completo

Matriz de confusao do teste:

```text
[[20094,    47,   386],
 [ 1058, 51556,    54],
 [  230,    79, 29663]]
```

Classes:

- linha/coluna 0 = `Normal`
- linha/coluna 1 = `Flooding`
- linha/coluna 2 = `Intrusao`

### Leituras importantes

#### Erros de ataques como benignos

Ataques classificados como `Normal`:

- `Flooding -> Normal = 1058`
- `Intrusao -> Normal = 230`

Total:

- `1288`

Esse numero e importante porque representa o tipo de erro mais sensivel do ponto de vista de deteccao: ataque que passa como benigno.

#### Falsos alarmes sobre trafego normal

Normais classificados como ataque:

- `Normal -> Flooding = 47`
- `Normal -> Intrusao = 386`

Total:

- `433`

Isso representa trafego benigno sinalizado indevidamente.

### Interpretacao operacional

O baseline:

- deixa escapar mais ataques como `Normal` do que o modelo otimizado;
- mas dispara menos alarmes falsos sobre trafego benigno.

---

## 5.2 Modelo otimizado

Matriz de confusao do teste:

```text
[[19984,   367,   176],
 [  595, 51982,    91],
 [  589,   127, 29256]]
```

### Leituras importantes

#### Erros de ataques como benignos

Ataques classificados como `Normal`:

- `Flooding -> Normal = 595`
- `Intrusao -> Normal = 589`

Total:

- `1184`

Aqui houve melhora em relacao ao baseline:

- `1184` em vez de `1288`

Ou seja:

- o tuning reduziu a quantidade total de ataques que passaram como benignos.

#### Falsos alarmes sobre trafego normal

Normais classificados como ataque:

- `Normal -> Flooding = 367`
- `Normal -> Intrusao = 176`

Total:

- `543`

Aqui houve piora em relacao ao baseline:

- `543` em vez de `433`

### Interpretacao operacional

O otimizado trocou:

- menos ataques classificados como normais

por:

- mais trafego normal classificado como ataque
- e pior recall para a classe `Intrusao`

Isso explica por que ele nao melhorou as metricas macro gerais.

---

## 5.3 Analise por classe

## Baseline

### `Normal`

- precision: `0.939762`
- recall: `0.978906`

Interpretacao:

- quando o baseline diz `Normal`, ele geralmente acerta;
- e ele reconhece quase todos os normais reais.

### `Flooding`

- precision: `0.997562`
- recall: `0.978887`

Interpretacao:

- classe extremamente bem aprendida;
- pouquissimos falsos positivos;
- excelente cobertura.

### `Intrusao`

- precision: `0.985384`
- recall: `0.989690`

Interpretacao:

- classe tambem muito bem modelada;
- o baseline esta muito forte aqui.

## Otimizado

### `Normal`

- precision: `0.944067`
- recall: `0.973547`

Interpretacao:

- ligeiro ganho de precision;
- pequena perda de recall.

### `Flooding`

- precision: `0.990586`
- recall: `0.986975`

Interpretacao:

- melhor recall de flooding;
- pior precision do que o baseline.

### `Intrusao`

- precision: `0.990956`
- recall: `0.976111`

Interpretacao:

- a precision subiu;
- mas o recall caiu perceptivelmente.

Ou seja:

- o otimizado ficou mais conservador ao chamar algo de `Intrusao`;
- isso reduziu algumas confusoes, mas deixou mais intrusoes passarem.

---

## 6. O que os Graficos de Generalizacao Dizem

## 6.1 Gaps treino vs teste

### Baseline completo

Do `generalization_report_baseline.json`:

- `accuracy_gap = -0.00114`
- `balanced_accuracy_gap = -0.00129`
- `f1_macro_gap = -0.00150`
- `mcc_gap = -0.00182`

### Modelo otimizado

- `accuracy_gap = -0.00107`
- `balanced_accuracy_gap = -0.00092`
- `f1_macro_gap = -0.00131`
- `mcc_gap = -0.00171`

### Como interpretar sinal negativo aqui

O gap foi calculado como:

```text
train - test
```

Mas em ambos os casos o teste ficou ligeiramente melhor que o treino em algumas metricas.

Isso nao e um problema.

Pelo contrario:

- indica ausencia de memorizacao excessiva;
- sugere que o treino nao ficou artificialmente inflado;
- e coerente com pequenas variacoes estatisticas normais.

### Conclusao

Os gaps sao minimos.

Isso e forte evidencia de:

- boa generalizacao;
- ausencia de overfitting relevante.

---

## 6.2 Curvas de aprendizado

### Baseline

A curva de aprendizado do baseline mostra:

- treino e validacao muito proximos;
- ambos melhorando com mais dados;
- estabilizacao em patamar alto;
- intervalo de incerteza pequeno.

### O que isso significa

Se o modelo estivesse decorando:

- a curva de treino ficaria muito acima da curva de validacao.

Se o modelo estivesse subajustado:

- ambas as curvas ficariam baixas e se estabilizariam cedo.

O que vemos aqui e:

- curvas altas;
- gap pequeno;
- melhora consistente com aumento do conjunto de treino.

Esse e o comportamento esperado de um modelo que aprendeu padroes reais.

### Otimizado

A curva do otimizado tem comportamento semelhante, tambem com gap pequeno.

Isso indica que o tuning nao gerou sobreajuste aparente.

---

## 6.3 Curva de convergencia

O `loss_curve_baseline.png` mostra:

- perda de treino caindo rapidamente;
- score de validacao interna alto e estavel;
- ausencia de explosao ou oscilacao caotica.

### O que isso diz

O MLP:

- convergiu;
- nao divergiu numericamente;
- nao mostra comportamento de instabilidade de treino.

Isso reforca que:

- o escalonamento foi apropriado;
- a arquitetura esta razoavelmente adequada ao problema.

---

## 7. Houve Overfitting?

## 7.1 Resposta curta

Nao ha evidencia forte de overfitting relevante.

## 7.2 Justificativa

Temos varios sinais consistentes:

1. Gaps treino/teste muito pequenos.
2. Curvas de aprendizado com gap pequeno.
3. Validacao cruzada limpa com score alto e estavel.
4. Teste tao bom quanto treino, ou ligeiramente melhor em alguns casos.
5. Matriz de confusao coerente, sem colapso em uma classe.

### O que seria um sinal de overfitting aqui?

- treino muito melhor que teste;
- AUC quase perfeita no treino e queda forte no teste;
- curva de treino subindo enquanto validacao estagna ou cai;
- grandes gaps em `F1 Macro` e `MCC`.

Nada disso aparece de forma forte nos resultados.

---

## 8. Houve Data Leakage?

## 8.1 Resposta curta

Nao ha indicio forte de data leakage na implementacao atual.

## 8.2 Por que podemos dizer isso

O pipeline foi desenhado justamente para evitar leakage:

1. O `train_test_split` ocorre antes de qualquer preprocessamento aprendido.
2. O imputador e ajustado apenas no treino.
3. O scaler e ajustado apenas no treino.
4. O `VarianceThreshold` e ajustado apenas no treino.
5. O `SMOTE` e aplicado apenas no treino.
6. A validacao cruzada reexecuta limpeza, selecao, escalonamento e SMOTE dentro de cada dobra.
7. Features de identificacao de ambiente foram removidas, reduzindo risco de "leakage semantico".

## 8.3 O que poderia denunciar leakage

Sinais suspeitos seriam:

- desempenho absurdamente perfeito sem justificativa;
- treino muito superior a CV;
- CV e teste artificialmente altos com diferencas quase impossiveis;
- uso de IPs, timestamps e identificadores na modelagem.

Como esses atalhos foram explicitamente removidos e as curvas estao saudaveis, nao vemos evidencias de leakage.

---

## 9. O Modelo Aprendeu ou Decorou?

## 9.1 Resposta curta

Os resultados sugerem fortemente que o modelo **aprendeu padroes reais**, e nao apenas decorou.

## 9.2 Por que isso parece aprendizado real

### 1. Features com sentido de dominio

As features mais importantes sao:

- `Protocol`
- `Pkt Len Var`
- `Bwd IAT Min`
- `Pkt Len Mean`
- `SYN Flag Cnt`
- `Pkt Len Std`
- `Flow IAT Mean`
- `Flow Pkts/s`

Todas elas tem relacao direta com o comportamento esperado de:

- flooding;
- reconhecimento;
- intrusao;
- assimetria de trafego;
- temporizacao.

Se o modelo estivesse decorando o ambiente, esperaríamos ver maior dependencia de:

- IP;
- porta;
- timestamp;
- flow id;
- campos mais ligados ao laboratorio do que ao comportamento.

Esses atributos nao estao sendo usados.

### 2. Generalizacao fora do treino

O desempenho no teste e do mesmo nivel do treino.

Memorizacao geralmente gera:

- treino muito alto;
- teste perceptivelmente pior.

Nao e o caso aqui.

### 3. Validacao cruzada limpa

As metricas de CV tambem sao altas:

#### Baseline completo

- `accuracy`: `0.979686`
- `balanced_accuracy`: `0.979424`
- `f1_macro`: `0.975399`

Isso mostra que o comportamento se repete em diferentes subconjuntos do treino.

### 4. Curvas de aprendizado

As curvas melhoram com mais dados e estabilizam sem explosao de gap.

Isso e tipico de aprendizagem efetiva.

---

## 10. O Modelo se Saiu Bem?

## 10.1 Resposta curta

Sim, o modelo se saiu muito bem.

## 10.2 Justificativa

Para classificacao multiclasse em problema realista de trafego:

- `F1 Macro ~ 0.978` no baseline completo e excelente;
- `Balanced Accuracy ~ 0.982` e excelente;
- `MCC ~ 0.971` e excelente;
- `ROC-AUC ~ 0.999` e extremamente alto;
- erro estrutural nas classes e baixo.

### O que isso significa na pratica

O sistema:

- distingue muito bem `Flooding`;
- distingue muito bem `Intrusao`;
- ainda lida bem com `Normal`;
- mantem baixo risco de viés excessivo para uma unica classe.

## 10.3 O que ainda merece atencao

O tipo de erro mais critico continua sendo:

- ataques previstos como `Normal`

No baseline completo isso foi:

- `1288` casos

No otimizado:

- `1184` casos

Ou seja:

- existe margem para melhora, especialmente se o objetivo operacional priorizar recall de ataque acima de tudo.

---

## 11. Baseline vs Tuning: o que aconteceu?

## 11.1 O tuning melhorou?

No agregado, nao.

O baseline venceu em:

- `Accuracy`
- `Balanced Accuracy`
- `F1 Macro`
- `F1 Weighted`
- `MCC`
- `G-Mean`
- `ROC-AUC`

O otimizado venceu apenas levemente em:

- `Precision Macro`

## 11.2 Como interpretar isso

O tuning mudou o trade-off:

- reduziu ataques previstos como `Normal`;
- aumentou falsos alarmes sobre `Normal`;
- reduziu recall de `Intrusao`;
- deixou o comportamento global um pouco pior.

### Conclusao

O tuning nao foi inútil; ele revelou que:

- o baseline ja estava muito bem posicionado;
- mexer na arquitetura deslocou a fronteira de decisao, mas nao trouxe ganho agregado.

Isso e uma descoberta valida.

---

## 12. O que os Resultados Dizem sobre o Problema

Esses resultados sugerem que o problema:

- nao e trivial;
- mas e fortemente estruturado;
- possui boa separabilidade pratica no espaco de features escolhido.

`Flooding` parece particularmente bem separavel.

`Intrusao`, apesar de heterogenea, tambem foi aprendida com excelente desempenho.

Isso indica que:

- os atributos de fluxo selecionados conseguem capturar o comportamento das classes;
- o agrupamento de labels fez sentido do ponto de vista estatistico e operacional.

---

## 13. Conclusao Final

Os resultados atuais apontam para um pipeline metodologicamente solido e um modelo que realmente aprendeu padroes relevantes do problema.

### Conclusoes principais

1. As metricas escolhidas sao adequadas porque o problema e multiclasse e desbalanceado.
2. Olhar apenas `Accuracy` seria insuficiente; por isso `Balanced Accuracy`, `F1 Macro`, `MCC` e `G-Mean` sao essenciais.
3. O baseline completo e o melhor modelo observado ate agora.
4. O tuning nao trouxe ganho global, embora tenha alterado o trade-off entre falsos positivos e falsos negativos.
5. Nao ha sinal forte de overfitting.
6. Nao ha indicio forte de data leakage.
7. O comportamento das curvas e das matrizes de confusao sugere aprendizado real, e nao mera memorizacao.

### Leitura operacional mais importante

Se o objetivo principal for:

- minimizar falsos alarmes sobre trafego benigno

o baseline e melhor.

Se o objetivo principal for:

- reduzir o numero total de ataques previstos como `Normal`

o tuning trouxe pequena melhora nesse ponto especifico, mas com custo em outras metricas.

No panorama geral, o baseline continua sendo a escolha tecnicamente mais equilibrada.
