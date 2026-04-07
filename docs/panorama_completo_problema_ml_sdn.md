# Panorama Completo do Problema de ML em SDN

## 1. Introducao

Este documento apresenta um panorama completo do problema tratado pelo modulo de Machine Learning deste projeto, desde a origem do problema ate a implementacao da solucao atual.

O objetivo aqui nao e apenas descrever o codigo, mas explicar:

- qual problema real queremos resolver;
- por que esse problema existe em redes SDN;
- como o dataset foi gerado;
- como o problema foi formulado como tarefa de classificacao;
- se o problema parece linearmente separavel ou nao;
- quais atributos foram mantidos e removidos;
- como foi feita a engenharia de label;
- quais modelos, tecnicas e ferramentas foram utilizados;
- por que essas escolhas foram feitas;
- como interpretar a solucao final do ponto de vista conceitual.

---

## 2. Origem do Problema

## 2.1 O contexto de SDN

SDN, ou *Software-Defined Networking*, e um paradigma de rede em que o plano de controle e separado do plano de dados.

Em termos práticos:

- os dispositivos de rede deixam de concentrar toda a inteligencia localmente;
- o controle da rede fica centralizado em um controlador;
- as regras de encaminhamento podem ser geridas de forma programatica.

Isso traz varias vantagens:

- maior flexibilidade;
- reconfiguracao mais simples;
- melhor visibilidade global;
- automacao.

Mas essa centralizacao tambem cria novos riscos:

- o controlador se torna um ponto critico;
- fluxos maliciosos podem sobrecarregar a infraestrutura;
- ataques comuns e ataques especificos de SDN podem coexistir;
- dispositivos ou hosts internos comprometidos podem se tornar base para novas acoes maliciosas.

Em resumo, o mesmo mecanismo que traz observabilidade para a rede tambem cria novas superficies de ataque.

---

## 2.2 O problema de seguranca

O problema de seguranca aqui e detectar trafego malicioso em um ambiente SDN a partir de estatisticas de fluxo.

Mais especificamente, queremos distinguir:

- trafego normal;
- trafego de flooding;
- trafego relacionado a intrusao.

Esse problema e importante porque:

- flooding pode derrubar servicos ou sobrecarregar controlador/switches;
- intrusoes podem explorar vulnerabilidades, fazer reconhecimento, comprometer hosts e preparar ataques posteriores;
- a detecao precoce ajuda a reduzir impacto operacional e melhora resposta a incidentes.

---

## 2.3 O problema de ML

Do ponto de vista de aprendizado de maquina, o problema foi formulado como:

- **classificacao supervisionada multiclasse**

Em outras palavras:

- cada fluxo de rede e representado por um vetor de atributos;
- esse vetor recebe uma classe alvo;
- o modelo aprende um mapeamento `X -> y`.

### Entrada

Um fluxo resumido por estatisticas numéricas e indicadores de protocolo/flags.

### Saida

Uma entre tres classes:

- `Normal`
- `Flooding`
- `Intrusao`

---

## 3. Origem dos Dados

## 3.1 O dataset utilizado

O dataset utilizado e o `InSDN`, armazenado localmente em:

- `dataset/InSDN_DatasetCSV`

Os arquivos usados sao:

- `Normal_data.csv`
- `OVS.csv`
- `metasploitable-2.csv`

Esse dataset deriva do trabalho descrito em:

- [InSDN_A_Novel_SDN_Intrusion_Dataset.pdf](/home/jv/sdn_ml_ddos_detection/cientific_base/InSDN_A_Novel_SDN_Intrusion_Dataset.pdf)

---

## 3.2 Como o dataset foi gerado

Segundo o artigo, o ambiente experimental foi montado com:

- uma VM com Kali Linux representando atacante;
- uma VM com ONOS como controlador SDN;
- uma VM com OVS e Mininet;
- uma VM com Metasploitable 2 representando servicos vulneraveis.

O trafego gerado inclui:

- trafego normal de servicos reais;
- ataques DoS;
- ataques DDoS;
- password guessing;
- web attacks;
- probe/scanning;
- botnet;
- U2R/exploitation.

Esse ponto e importante: o dataset nao foi criado para representar apenas DDoS. Ele foi desenhado para representar um conjunto mais amplo de comportamentos maliciosos em SDN.

---

## 3.3 Tipo de dado

Os CSVs nao armazenam pacotes brutos, mas **estatisticas de fluxo**.

Isso significa que cada linha representa um fluxo resumido por medidas como:

- duracao;
- taxas de bytes/pacotes;
- tempos entre pacotes;
- contagem de pacotes por direcao;
- estatisticas de tamanho de pacote;
- contagem de flags TCP.

Essas features foram derivadas com uma ferramenta de extracao de fluxo, e nao manualmente.

Logo, este projeto trabalha com:

- **aprendizado tabular sobre atributos de fluxo**

e nao com:

- analise de payload;
- analise de pacotes crus;
- sequencias temporais profundas;
- embeddings de texto ou bytes.

---

## 4. Formulação do Problema

## 4.1 Labels originais

Os labels originais encontrados no dataset incluem:

- `Normal`
- `DoS`
- `DDoS`
- `Probe`
- `BFA`
- `Web-Attack`
- `BOTNET`
- `U2R`

---

## 4.2 Engenharia de label

O projeto atual **nao fez engenharia de feature ao criar a classe `Intrusao`**.

O que foi feito foi:

- **engenharia de label**
- **agrupamento da variavel alvo**
- **redefinicao semantica do target**

Mais precisamente, a coluna `Label` original foi remapeada para:

- `Normal` = `Normal`
- `Flooding` = `DoS + DDoS`
- `Intrusao` = `Probe + BFA + Web-Attack + BOTNET + U2R`

### Por que isso foi feito?

Porque o problema original com muitas classes:

- aumenta heterogeneidade;
- traz classes muito pequenas;
- torna o problema menos robusto para o objetivo prático do projeto.

Exemplos:

- `BOTNET` tem poucas amostras;
- `U2R` e extremamente raro;
- algumas classes sao semanticamente proximas do ponto de vista operacional.

Ao agrupar as classes, o problema se torna:

- mais coerente com o objetivo da aplicacao;
- mais robusto estatisticamente;
- menos fragmentado;
- mais viavel para um baseline confiavel.

### O que isso significa semanticamente?

- `Flooding` representa ataques cujo principal comportamento e saturacao/flood;
- `Intrusao` representa atividades maliciosas mais ligadas a reconhecimento, comprometimento, exploração, abuso de credenciais e controle indevido.

---

## 4.3 O que seria engenharia de feature, e o que nao foi feito aqui

Engenharia de feature seria:

- criar um novo atributo de entrada a partir de outros;
- transformar ou combinar colunas de `X`;
- introduzir atributos derivados.

Exemplos de engenharia de feature seriam:

- `bytes_por_pacote = Flow Byts/s / Flow Pkts/s`
- criar indicador de burst;
- log-transform de variaveis muito assimetricas;
- combinacoes de `IAT`.

Neste projeto, o que se fez foi:

- selecao de features;
- limpeza;
- escalonamento;
- **engenharia de label**, nao engenharia de feature.

---

## 5. O Problema e Linearmente Separavel?

## 5.1 Resposta curta

Provavelmente **nao e estritamente linearmente separavel**, mas parece ser **bem separavel** no espaco de atributos escolhido.

---

## 5.2 Por que nao assumir separabilidade linear?

Temos motivos para nao assumir isso:

- as classes agrupadas sao heterogeneas, especialmente `Intrusao`;
- os padroes de trafego dependem de combinacoes nao lineares de taxa, tempo, flags e assimetria;
- ataques diferentes podem compartilhar partes do mesmo padrao estatistico;
- algumas fronteiras de decisao entre `Normal` e `Intrusao` provavelmente sao irregulares.

Uma separacao puramente linear implicaria que um hiperplano simples resolveria bem o problema em todo o espaco.

Isso e improvavel em um problema que mistura:

- scanning;
- brute force;
- botnet;
- exploracao;
- flooding.

---

## 5.3 Por que dizemos que ele parece bem separavel?

Pelos resultados obtidos:

- `F1 Macro` alto;
- `Balanced Accuracy` alta;
- `MCC` alto;
- pequena diferenca entre treino e teste;
- matriz de confusao com poucos erros relativos;
- feature importance consistente e dominante em algumas variaveis fortes.

Esses sinais indicam que:

- o problema nao e caotico;
- as classes ocupam regioes relativamente distintas no espaco de atributos;
- mesmo sem tuning agressivo, o baseline ja performa muito bem.

### Conclusao tecnica

O problema parece:

- **nao linear em sentido estrito**
- mas **fortemente estruturado e razoavelmente separavel**

Em outras palavras:

- um modelo simples demais poderia perder padroes;
- um modelo com capacidade moderada, como um MLP pequeno, consegue capturar bem a estrutura.

---

## 6. Por que usar Estatisticas de Fluxo?

A escolha por atributos de fluxo, em vez de payload ou pacotes crus, tem motivos fortes:

1. Sao compativeis com monitoramento SDN.
2. Custam menos computacionalmente que analise profunda de pacotes.
3. Capturam ritmo, volume, regularidade e assimetria de trafego.
4. Sao suficientes para detectar muitos ataques comportamentais.

Em especial:

- flooding altera ritmo e volume;
- scanning altera padrao de tentativas;
- brute force muda periodicidade e repeticao;
- intrusoes podem gerar assimetria incomum e sequencias especificas de fluxo.

---

## 7. Atributos Considerados no Dataset

O dataset original possui muito mais do que as 26 features finais usadas no treino.

Ele inclui, entre outros:

- identificadores de fluxo;
- enderecos IP;
- portas;
- timestamp;
- contagens por direcao;
- tempos entre pacotes;
- medidas de tamanho de pacote;
- flags TCP;
- medidas de atividade e inatividade;
- medidas relacionadas a janela/segmentos;
- atributos de bulk/subflow.

Nem tudo isso foi usado.

---

## 8. Atributos Removidos e Por Que

A implementacao atual usa uma lista fixa de features em `RELEVANT_FEATURES`.

Tudo que ficou fora dessa lista, na pratica, foi removido do treinamento.

As remocoes podem ser entendidas em categorias.

## 8.1 Identificadores e metadados de ambiente

Removidos por risco alto de memorizacao:

- `Flow ID`
- `Src IP`
- `Dst IP`
- `Src Port`
- `Dst Port`
- `Timestamp`

### Motivo

Essas colunas podem fazer o modelo aprender o laboratorio e nao o comportamento do ataque.

Exemplo:

- se certo IP aparece sempre em ataque no dataset, o modelo pode decorar esse IP;
- isso funciona no laboratorio, mas falha fora dele.

Esse tipo de aprendizado e espurio do ponto de vista de generalizacao.

---

## 8.2 Features menos confiaveis para generalizacao operacional

Removidas por terem maior risco de dependencia forte do cenario:

- medidas muito especificas de janela/segmento inicial;
- medidas bulk/subflow pouco informativas no contexto atual;
- atributos redundantes diante das features mais centrais de taxa, tempo e tamanho.

Exemplos de colunas nao utilizadas:

- `Fwd Pkt Len Max`
- `Fwd Pkt Len Min`
- `Fwd Pkt Len Mean`
- `Bwd Pkt Len Max`
- `Pkt Len Min`
- `Pkt Len Max`
- `Fwd Header Len`
- `Bwd Header Len`
- `Fwd Pkts/s`
- `Pkt Size Avg`
- `Fwd Seg Size Avg`
- `Bwd Seg Size Avg`
- `Subflow Fwd Pkts`
- `Subflow Fwd Byts`
- `Subflow Bwd Pkts`
- `Subflow Bwd Byts`
- `Init Fwd Win Byts`
- `Init Bwd Win Byts`
- `Fwd Act Data Pkts`
- e outras nao listadas em `RELEVANT_FEATURES`

### Motivo

Nao necessariamente sao "ruins", mas:

- nao eram essenciais para o baseline;
- aumentariam dimensionalidade;
- podem aumentar chance de ruído, redundancia ou sobreajuste.

---

## 8.3 Features removiveis por baixa utilidade estatistica

Algumas features do dataset completo sao conhecidamente:

- constantes;
- quase constantes;
- ou muito pouco informativas em certas amostras.

No pipeline, isso e controlado por:

- `VarianceThreshold`

No experimento documentado, nenhuma das 26 features escolhidas foi removida por variancia zero, mas essa etapa existe para proteger o pipeline.

---

## 9. Atributos Finais Usados no Treinamento

As 26 features utilizadas foram:

- `Protocol`
- `Flow Duration`
- `Tot Fwd Pkts`
- `Tot Bwd Pkts`
- `TotLen Fwd Pkts`
- `TotLen Bwd Pkts`
- `Flow Byts/s`
- `Flow Pkts/s`
- `Flow IAT Mean`
- `Flow IAT Std`
- `Flow IAT Max`
- `Flow IAT Min`
- `Bwd IAT Tot`
- `Bwd IAT Mean`
- `Bwd IAT Std`
- `Bwd IAT Max`
- `Bwd IAT Min`
- `Bwd Pkts/s`
- `Pkt Len Mean`
- `Pkt Len Std`
- `Pkt Len Var`
- `Down/Up Ratio`
- `SYN Flag Cnt`
- `ACK Flag Cnt`
- `Active Mean`
- `Idle Mean`

---

## 9.1 Significado conceitual de cada grupo de atributos

### Protocolo

- `Protocol`

Ajuda a distinguir o tipo de comunicacao usada no fluxo.

### Duracao e ritmo

- `Flow Duration`
- `Flow Pkts/s`
- `Flow Byts/s`

Capturam velocidade e persistencia do fluxo.

### Contagem e volume

- `Tot Fwd Pkts`
- `Tot Bwd Pkts`
- `TotLen Fwd Pkts`
- `TotLen Bwd Pkts`

Capturam intensidade e assimetria de trafego.

### Temporizacao

- `Flow IAT Mean`
- `Flow IAT Std`
- `Flow IAT Max`
- `Flow IAT Min`
- `Bwd IAT Tot`
- `Bwd IAT Mean`
- `Bwd IAT Std`
- `Bwd IAT Max`
- `Bwd IAT Min`

Capturam regularidade, burst e espacamento temporal.

### Tamanho dos pacotes

- `Pkt Len Mean`
- `Pkt Len Std`
- `Pkt Len Var`

Capturam padronizacao ou variabilidade no tamanho dos pacotes.

### Assimetria e direcionalidade

- `Down/Up Ratio`
- `Bwd Pkts/s`

Mostram relacao entre ida e volta.

### Flags TCP

- `SYN Flag Cnt`
- `ACK Flag Cnt`

Muito uteis para padroes de estabelecimento de conexao e flooding TCP.

### Comportamento ativo/ocioso

- `Active Mean`
- `Idle Mean`

Descrevem alternancia entre atividade e pausas.

---

## 10. Comportamento Esperado das Classes

## 10.1 Normal

Tende a apresentar:

- maior diversidade de tamanhos;
- interacoes mais organicamente distribuidas no tempo;
- menos padroes extremos de flags;
- maior variabilidade de protocolos e uso.

## 10.2 Flooding

Tende a apresentar:

- taxas muito altas;
- pacotes mais padronizados;
- temporalidade regular ou em rajada;
- padroes fortes de protocolo/flag.

## 10.3 Intrusao

Tende a apresentar:

- interacao menos parecida com uso benigno;
- padroes temporais e direcionais atipicos;
- mistura de comportamento de reconhecimento, exploração e abuso.

Essa classe e mais heterogenea, o que a torna conceitualmente mais dificil que `Flooding`.

---

## 11. Modelos e Ferramentas Utilizados

## 11.1 Ferramentas de desenvolvimento e analise

- `pandas`
  leitura e manipulacao dos dados.

- `numpy`
  operacoes numericas.

- `scikit-learn`
  split, imputacao, escalonamento, MLP, validacao e metricas.

- `imbalanced-learn`
  `SMOTE`.

- `matplotlib`
  geracao de graficos.

- `joblib`
  persistencia dos artefatos.

- `shap`
  interpretabilidade e importancia de features.

---

## 11.2 Modelo principal

Foi escolhido um:

- `MLPClassifier`

### Motivos

1. O problema e tabular e supervisionado.
2. As fronteiras de decisao provavelmente nao sao puramente lineares.
3. O MLP consegue capturar interacoes nao lineares sem a complexidade de modelos profundos maiores.
4. O custo computacional continua razoavel.
5. O baseline e forte o suficiente para medir separabilidade do problema.

### Por que nao regressao logistica?

Porque a regressao logistica impõe fronteiras lineares. Como o problema mistura classes heterogeneas, ela poderia ficar limitada.

### Por que nao CNN ou RNN?

Porque os dados nao sao imagem nem sequencia crua; sao vetores tabulares de fluxo.

### Por que nao so RandomForest?

A RandomForest foi usada como auxiliar de interpretacao/importancia, nao como classificador final. O objetivo do projeto atual estava centrado no MLP.

---

## 12. Pipeline de Dados: Como o Problema e Resolvido na Pratica

## 12.1 Carregamento

O loader:

- concatena os CSVs;
- normaliza strings de label;
- remapeia as classes;
- seleciona as colunas desejadas;
- produz `X` e `y`.

## 12.2 Split

O dataset e dividido em:

- treino
- teste

usando:

- `train_test_split`
- `stratify=y`

Isso preserva a distribuicao das classes.

## 12.3 Limpeza

O `DataCleaner`:

- converte valores negativos impossiveis em `NaN`;
- substitui `Inf` por `NaN`;
- imputa com mediana;
- remove duplicatas reais.

## 12.4 Selecao de features

O `FeatureSelector`:

- remove colunas constantes;
- calcula importancia por SHAP;
- produz ranking para auditoria.

## 12.5 Escalonamento

O `FeatureScaler`:

- ajusta `StandardScaler` no treino;
- preserva features binarias conhecidas sem padronizacao;
- transforma treino e teste com o mesmo mapeamento.

## 12.6 Balanceamento

O `ClassBalancer` usa:

- `SMOTE`

para equilibrar as classes no treino.

## 12.7 Treinamento

O `ModelTrainer`:

- treina o MLP baseline;
- salva curva de convergencia;
- executa validacao cruzada limpa.

## 12.8 Avaliacao

O `ModelEvaluator`:

- calcula metricas multiclasse;
- gera matriz de confusao.

## 12.9 Diagnostico

O `TrainingDiagnostics`:

- gera curva de aprendizado;
- gera grafico de gap treino/teste;
- salva relatorio de generalizacao.

## 12.10 Persistencia

O `ModelIO` salva:

- modelo;
- imputador;
- seletor de variancia;
- scaler;
- lista de features.

---

## 13. Validação Cruzada Limpa

Um ponto importante da implementacao atual e que a validacao cruzada nao usa apenas o modelo treinado sobre dados ja preprocessados globalmente.

Ela reaplica o pipeline inteiro dentro de cada dobra:

- limpeza;
- selecao;
- escalonamento;
- SMOTE;
- treino.

Isso e metodologicamente mais correto, porque evita vazamento da dobra de validacao para as transformacoes.

---

## 14. Overfitting e Generalizacao

O projeto nao se limita a relatar metrica final.

Ele monitora:

- curva de loss;
- curva de aprendizado;
- comparacao treino vs teste;
- gaps numericos entre treino e teste.

### Como interpretamos isso?

Se:

- treino muito maior que teste;
- curva de treino sobe mas validacao estagna;
- gap alto em `F1 Macro`, `MCC` ou `Balanced Accuracy`

entao ha suspeita de overfitting.

Se os gaps sao pequenos e as curvas convergem:

- o modelo generaliza melhor.

---

## 15. O que os Resultados Sugerem sobre o Problema

Pelos experimentos realizados no projeto:

- o problema parece bem estruturado;
- a classe `Flooding` e particularmente facil de distinguir;
- `Intrusao` tambem fica bem separada, apesar da heterogeneidade;
- os erros mais sensiveis aparecem quando ataques sao confundidos com `Normal`.

Isso indica que:

- as features escolhidas capturam bem a fenomenologia do problema;
- o baseline MLP ja oferece um ponto de partida muito forte;
- o problema tem alta separabilidade pratica, embora nao devamos chamá-lo de linearmente separavel em sentido estrito.

---

## 16. Exemplo Conceitual de Como o Modelo Decide

Imagine tres fluxos:

### Exemplo A: fluxo normal

- taxa moderada;
- tamanhos de pacote variados;
- tempos entre pacotes mais irregulares;
- sem contagem anormal de `SYN`.

O modelo tende a classificar como:

- `Normal`

### Exemplo B: flooding TCP/UDP

- `Flow Pkts/s` muito alto;
- `SYN Flag Cnt` alto ou padrao forte de protocolo;
- baixa variancia de tamanho de pacote;
- `IAT` muito curto e regular.

O modelo tende a classificar como:

- `Flooding`

### Exemplo C: tentativa de probe/brute force/exploracao

- padrao de ida e volta estranho;
- duracao e temporizacao atipicas;
- fluxo nao necessariamente volumoso, mas estatisticamente diferente de uso benigno.

O modelo tende a classificar como:

- `Intrusao`

---

## 17. Limitações Conceituais Atuais

Mesmo com boa implementacao, ha limitacoes:

1. `Intrusao` e uma classe agregada e heterogenea.
2. O agrupamento simplifica a semantica original do dataset.
3. Um bom desempenho nao garante robustez absoluta fora do cenario do dataset.
4. O baseline ainda nao esgota o espaco de modelos possiveis.

Essas limitacoes sao normais e nao invalidam a abordagem. Apenas delimitam o que o sistema significa.

---

## 18. Conclusao Geral

O problema tratado neste projeto nasce da necessidade de detectar trafego malicioso em um ambiente SDN usando atributos de fluxo.

O dataset InSDN oferece uma base adequada porque:

- foi construido em um ambiente SDN;
- contempla multiplas familias de ataques;
- permite modelagem tabular por comportamento de fluxo.

A solucao atual:

- redefiniu o alvo por engenharia de label;
- evitou atributos com alto risco de memorizacao;
- escolheu atributos comportamentais de fluxo;
- usou um MLP para capturar fronteiras nao lineares;
- aplicou um pipeline metodologicamente correto;
- produziu uma solucao forte e interpretavel.

Em resumo:

- o problema nao deve ser tratado como puramente linear;
- mas ele e suficientemente estruturado para ser bem resolvido por um MLP pequeno com preprocessamento adequado;
- a formulacao `Normal / Flooding / Intrusao` representa um equilibrio entre fidelidade ao dataset, viabilidade estatistica e utilidade pratica.
