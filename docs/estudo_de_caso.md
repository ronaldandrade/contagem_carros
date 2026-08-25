# Contagem de Carros

Projeto de contagem de veículos em fluxo de trânsito.
O vídeo usado foi gravado de propósito em um ângulo ruim, com árvores na frente
da rua, para testar até onde a técnica funciona em uma situação pouco favorável.
Ambiente perfeito não cria boa solução.

Na prática, o projeto acabou virando um estudo de três eixos:

1. **Visão computacional sob restrição** — quanto um detector aguenta de oclusão,
   distância e ângulo ruim, e quanto treinar na própria cena recupera.
2. **Honestidade metodológica** — o primeiro resultado foi bom demais e era falso.
   Diagnosticar e corrigir o vazamento treino/validação virou a parte mais útil
   do trabalho.
3. **Uso de recursos de hardware** — quanto de velocidade dá para extrair da GPU
   levando o modelo de PyTorch a TensorRT, e o que isso custa em acurácia e
   tamanho.

## O que o projeto faz

1. Detecta e acompanha os veículos no vídeo.
2. Conta um veículo quando ele cruza uma linha virtual desenhada no quadro,
   guardando horário, tipo (carro, moto, ônibus, caminhão) e sentido
   (subindo ou descendo) em um arquivo CSV.
3. Compara as gravações de horários diferentes e mostra em gráfico qual é o
   horário de maior e de menor movimento, e de que tipos de veículo o trânsito é
   formado.
4. Mede a qualidade da detecção comparando o que o modelo encontrou com caixas
   marcadas à mão em uma amostra de frames.
5. Treina um modelo próprio com essas mesmas imagens e compara com o modelo
   genérico.
6. Refaz a divisão treino/validação por bloco de tempo, para medir o quanto o
   resultado anterior estava inflado por vazamento.
7. Exporta o modelo para ONNX e TensorRT (FP16 e INT8) e mede o trade-off entre
   velocidade, acurácia e tamanho em cada formato.

## Como funciona

A detecção usa o *YOLOv8*, que aponta onde estão os veículos em cada quadro do
vídeo. Como o *YOLO* não sabe que o carro do quadro 10 é o mesmo do quadro 11, o
*ByteTrack* entra em seguida e dá um identificador para cada veículo, mantendo esse
identificador enquanto ele aparece na cena.

A contagem em si é simples: existe uma linha virtual na imagem, definida em
posição relativa para funcionar em qualquer resolução. Para cada veículo, o
programa olha de que lado da linha está o centro da caixa. Quando o lado muda, o
veículo cruzou a linha e é contado uma única vez, mesmo que continue aparecendo
depois. O sentido vem justamente de qual lado ele estava antes.

O fluxo completo, do vídeo à métrica:

```
vídeo  ──►  extrair_frames.py  ──►  frames/*.jpg
                                         │
                             anotação manual (Roboflow, CVAT, LabelImg…)
                                         │
                                    labels/*.txt  (formato YOLO)
                                         │
                              preparar_split.py  (corte temporal + tampão)
                                         │
frames + labels + modelo  ──►  avaliar_detector.py  ──►  AP por classe, mAP,
                                         │                curva P-R, melhor limiar
                                    nucleo_avaliacao.py  (a matemática, testável)
```

Os arquivos:

- `src/extrair_frames.py` retira frames do vídeo, espalhados de forma uniforme,
  para serem marcados à mão.
- `src/contagem_carros.py` faz a detecção, o rastreamento e a contagem, gerando o
  CSV.
- `src/analise_trafego.py` lê o CSV e gera o gráfico de comparação entre
  horários.
- `src/nucleo_avaliacao.py` tem as contas da avaliação, separadas do resto para
  poderem ser testadas sozinhas. Rodando o arquivo direto, ele confere as contas
  em três casos simples de verificar na mão.
- `src/deteccao.py` é a ponte YOLO → arrays, que mantém o núcleo livre de YOLO.
- `src/avaliar_detector.py` junta o modelo, as marcações manuais e as contas, e
  produz os gráficos e a tabela de resultados.
- `src/preparar_split.py` refaz a divisão treino/validação por bloco de tempo, com
  zona-tampão, corrigindo o vazamento do split aleatório original.
- `src/benchmark_formatos.py` leva o modelo por PyTorch → ONNX → TensorRT FP16 →
  TensorRT INT8 e mede velocidade, acurácia e tamanho em cada formato, gerando uma
  tabela reprodutível junto com o ambiente exato em que foi medida.

## Como usar

```bash
pip install -r requirements.txt

# contar os veículos de um vídeo, informando o horário da gravação
python src/contagem_carros.py --video data/raw/video.mp4 --janela "13:00"

# comparar os horários já contados
python src/analise_trafego.py --csv outputs/contagens.csv --minutos 10

# separar frames para marcar à mão
python src/extrair_frames.py --video data/raw/video.mp4 --n 60 --saida data/frames

# refazer a divisão treino/validação por bloco de tempo
python src/preparar_split.py --val-inicio 52 --tampao 4

# avaliar o modelo genérico
python src/avaliar_detector.py --imagens data/splits/temporal/images/val \
    --labels data/splits/temporal/labels/val --modelo models/yolov8n.pt --classes coco

# avaliar o modelo treinado neste dataset
python src/avaliar_detector.py --imagens data/splits/temporal/images/val \
    --labels data/splits/temporal/labels/val --modelo models/treino_original.pt --classes custom

# benchmark de formatos de deploy
python src/benchmark_formatos.py --modelo models/split_temporal.pt --classes custom
```

O autoteste das contas roda sozinho com `python src/nucleo_avaliacao.py`.

## O dataset

O dataset foi feito do zero a partir do próprio vídeo. Foram separados 60 frames
espalhados pela gravação e cada veículo foi marcado à mão, com uma caixa por
veículo, no formato do YOLO (um arquivo de texto por imagem, com a classe e a
posição da caixa em valores de 0 a 1: `classe cx cy w h`, todos normalizados entre
0 e 1).

São **163 caixas** marcadas ao todo — 134 carros, 28 ônibus e 1 moto. Para o treino,
as imagens foram separadas em 48 para treinar e 12 para validar — dataset
extremamente pequeno, mas proposital, a princípio, para evoluções no estudo mais à
frente.

### A lição sobre amostragem

Eu comecei acreditando que 60 a 100 frames já davam uma estimativa estável. Está
errado, e o motivo é o assunto central deste documento: **o que importa não é a
quantidade de frames, é o espaçamento entre eles e a variedade de gravações.**

O vídeo tem cerca de 310 segundos e eu tirei 60 frames — um a cada 5 segundos. Um
ônibus leva mais que isso para atravessar a cena, então o mesmo ônibus aparece em
vários frames e vira várias caixas anotadas. A amostra parece maior do que é: o
número de caixas anotadas não mede o tamanho da amostra, quem mede é a quantidade
de veículos *diferentes*, e essa não está anotada em lugar nenhum.

A regra melhor:

- espaçamento maior que o tempo que um veículo leva para cruzar o quadro (uns 15 a
  20 segundos nesta cena);
- frames vindos de várias gravações em vez de uma só — **20 frames de cinco vídeos
  diferentes valem mais que 100 frames de um vídeo**;
- sortear os frames ao acaso em vez de uniformemente **não resolve nada**, porque o
  espaçamento médio continua o mesmo e o sorteio ainda pode juntar dois frames
  separados por meio segundo.

### Cuidado com os IDs de classe

O script já resolve os dois casos e não é preciso reanotar nada, mas trocar a opção
por engano **zera o resultado**, porque a comparação passa a ser feita entre classes
diferentes:

- anotação exportada pelo Roboflow com as classes do `data.yaml`
  (`0=ônibus, 1=carro, 2=moto`) — é o caso deste projeto. Para avaliar o modelo
  genérico use `--classes coco` (o padrão) e o script traduz os IDs para os do COCO;
  para avaliar o modelo treinado neste dataset use `--classes custom`, porque aí o
  modelo já devolve os mesmos IDs da anotação e nenhuma tradução deve acontecer;
- anotação feita direto nos IDs do COCO (`2=carro, 3=moto, 5=ônibus, 7=caminhão`):
  use `--classes coco` e ajuste o dicionário `GT_IDENTIDADE`/`ROBOFLOW_PARA_COCO`
  em `avaliar_detector.py`.

## As medidas usadas

Antes de tudo é preciso decidir quando uma caixa prevista conta como acerto. A
régua é o **IoU** (*Intersection over Union*): a sobreposição entre a caixa prevista
e a caixa marcada à mão, dividida pela área total que as duas ocupam juntas.

$$
\text{IoU}(A, B) = \frac{\text{área}(A \cap B)}{\text{área}(A \cup B)}
$$

Vale 0 quando não se tocam e 1 quando são iguais. O padrão adotado aqui é
considerar acerto a partir de **0,5**.

### O casamento predição ↔ realidade

Detecção não vem rotulada como certa ou errada; é preciso **casar** cada predição
com uma caixa real (`casar_por_imagem`), pela regra padrão da área:

1. ordene as predições do frame por confiança, da maior para a menor;
2. para cada predição, na ordem, pegue a caixa real de maior IoU **ainda não usada**;
3. se esse IoU ≥ 0,5 → **verdadeiro positivo (TP)**, e aquela caixa real fica
   consumida;
4. senão → **falso positivo (FP)**.

O detalhe crucial é o "ainda não usada": cada veículo real só pode ser acertado uma
vez. Se o modelo desenha três caixas no mesmo carro, uma conta como TP e as outras
duas como FP — comportamento correto, e algo que o ângulo difícil provoca bastante.

### Precisão, recall, AP e mAP

$$
\text{Precisão} = \frac{TP}{TP + FP} \qquad \text{Recall} = \frac{TP}{\text{total de GT}}
$$

- **Precisão**: das caixas que o modelo desenhou, quantas estavam certas (penaliza
  alarme falso).
- **Recall**: dos veículos que existiam, quantos o modelo encontrou (penaliza o que
  passou batido).
- **F1**: a média harmônica das duas, $F_1 = 2PR/(P+R)$, um número único que
  equilibra os dois erros.
- **AP**: a área sob a curva precisão-recall. Ordenamos todas as predições por
  confiança e acumulamos TP e FP descendo a lista; cada posição vira um ponto
  (recall, precisão). Uso a interpolação *all-points* (padrão COCO): antes de
  integrar, torno a curva monótona pegando em cada ponto a maior precisão à direita.
- **mAP**: a média do AP entre as classes que têm pelo menos um exemplo real no
  conjunto.

Precisão e recall puxam para lados opostos. Exigir mais confiança do modelo faz
ele errar menos, mas deixar mais veículos passarem. Exigir menos faz o contrário.
Por isso o projeto gera uma tabela com essas medidas em cada nível de confiança e
aponta qual dá o melhor equilíbrio — o de maior F1. Esse é o valor que deveria ser
usado na contagem: a avaliação não serve só para dar uma nota, ela também calibra o
contador. (Ver a ressalva na seção *Limitações conhecidas do código*: hoje essa
ligação é manual.)

### Por que confiar nestas contas

`nucleo_avaliacao.py` não importa YOLO, não lê arquivos e não desenha nada — é só
numpy. Isso permite o autoteste, que confere a matemática do AP em três casos
verificáveis na mão: detecções perfeitas (AP = 1,000), um falso positivo no meio da
lista (AP ≈ 0,833) e nenhuma detecção (AP = 0,000). Se algum falhar, o `assert`
quebra. Separar a matemática da inferência é o que garante que o número publicado
está certo.

## Resultados

O projeto mede três coisas, e nenhuma delas é "quantos carros passaram":

1. **Um detector aguenta uma cena adversa?** A gravação tem árvores na frente da
   rua, ângulo ruim e veículos distantes. Foi de propósito.
2. **A metodologia sustenta o número que produz?** Esta acabou sendo a parte mais
   útil do trabalho, e a que mais deu errado antes de dar certo.
3. **Quanto do hardware dá para extrair?** Medido na seção seguinte.

Ambiente perfeito não cria boa solução. O interesse está justamente em medir sob
restrição — e em descobrir quando o próprio número medido está mentindo.

### A cena

O vídeo foi gravado com a câmera parada, em um ângulo desfavorável, com obstáculos
(árvores) entre a câmera e a rua. Veículos entram e saem de trás da folhagem,
aparecem parcialmente ocluídos e ocupam poucos pixels. É um cenário em que um
detector pronto de prateleira não deveria funcionar bem — e não funciona:

| nos 12 frames de validação | YOLOv8n genérico (COCO) |
|---|---|
| mAP@0,5 | 0,020 |
| melhor F1 | 0,065 |

Esse é o ponto de partida. A pergunta do projeto passa a ser: **treinar com as
imagens da própria cena recupera o que a cena adversa tirou?**

### Primeiro treino: split aleatório — e por que o resultado era falso

O primeiro modelo foi treinado dividindo os 60 frames anotados ao acaso: 48 para
treinar, 12 para validar. O resultado pareceu excelente:

| primeiro treino (split aleatório) | |
|---|---|
| mAP@0,5 | 0,660 |
| AP em carro | 0,974 |
| AP em ônibus | **1,000** |
| melhor F1 | 0,919 |

Nota 1,000 numa classe, com 60 frames, é bom demais para ser verdade — e era
motivo para investigar, não para comemorar. Frames vizinhos de um mesmo vídeo não
são exemplos independentes: o sorteio colocou o **mesmo veículo** nos dois lados da
divisão. O modelo não estava sendo avaliado, estava sendo consultado sobre o que
já tinha visto. Ele não reconhecia *um* ônibus, reconhecia *aquele* ônibus.

### O diagnóstico, em números

Três medições feitas nos próprios rótulos:

- **58% das caixas têm IoU ≥ 0,5 com uma caixa da mesma classe no frame anterior**
  (40% com IoU ≥ 0,7). É o mesmo veículo, quase na mesma posição.
- **Os 12 frames de validação tinham distância 1 para o treino. Todos os doze.**
  Não era um split parcialmente contaminado — não sobrava um frame limpo sequer.
- **As 28 caixas de "ônibus" são 2 ônibus:** um parado nos frames 7 a 34
  (27 caixas ininterruptas), mais um no frame 47. E há **1 única caixa de moto** no
  dataset inteiro.

Isso desmonta o `0,660`: ele era a média de `ônibus 1,000` (um ônibus decorado),
`carro 0,974` (inflado) e `moto 0,000` (uma instância nunca detectada). Nenhuma das
três parcelas media generalização. Vale notar o efeito colateral: como o mAP é a
média das classes, esse zero sozinho puxa a nota de perto de 0,99 para 0,660 — o
número reportado como resumo do modelo era, na prática, decidido por **uma única
moto**.

Importante: **os rótulos estavam certos.** Um ônibus marcado em 27 frames são 27
caixas corretas. O erro não foi de anotação, foi de *amostragem e divisão* — e por
isso a correção não exigiu re-anotar nada, nem custou nada além de refazer o corte.

### A nota muda conforme o conjunto onde se mede

Isso dá para demonstrar em vez de só afirmar. As mesmas duas avaliações, rodadas
nos 60 frames anotados e depois só nos 12 frames de validação:

| | genérico (60 frames) | genérico (12 de val.) | treinado (60 frames) | treinado (12 de val.) |
|---|---|---|---|---|
| mAP@0,5 | 0,060 | 0,020 | 0,660 | 0,660 |
| melhor F1 | 0,154 | 0,065 | 0,943 | 0,921 |

O modelo treinado vai melhor nos 60 frames porque 48 deles são exatamente as
imagens em que ele treinou — é a nota de uma prova com as respostas anotadas na
mesa. O 0,921 da validação é o menos ruim dos dois, e mesmo ele está contaminado
pelo vazamento.

### Segundo treino: split temporal com zona-tampão

`src/preparar_split.py` divide por bloco de tempo contíguo, descartando uma faixa
de frames na fronteira para que nenhum veículo apareça dos dois lados:

```
frames 0–47   |   48–51    |   52–65
  TREINO      |   TAMPÃO   |  VALIDAÇÃO
 47 frames    | descartado |  9 frames
```

```bash
python src/preparar_split.py --val-inicio 52 --tampao 4
```

A distância mínima entre validação e treino sobe de **1 para 5 frames** (~35 s) —
tempo de sobra para qualquer veículo em movimento sair de cena.

Para que a diferença fosse atribuível ao split e não ao treino, **os dois braços
foram retreinados do zero com hiperparâmetros idênticos** (yolov8n, 60 épocas,
batch 8, imgsz 640, seed 0, patience 20). Controle de sanidade: o braço aleatório
retreinado reproduz o modelo histórico (0,652 contra 0,660 — a diferença vem de
treinar em GPU em vez de CPU).

Comparação honesta só na classe **carro** — a única com exemplos dos dois lados
do corte:

| classe carro | split aleatório (vazado) | split temporal (limpo) | queda |
|---|---|---|---|
| AP@0,5 *(métrica do projeto)* | 0,974 | **0,853** | −0,121 |
| AP@0,5 *(métrica do Ultralytics)* | 0,960 | **0,881** | −0,079 |
| AP@0,5:0,95 | 0,502 | **0,402** | −0,100 |
| melhor F1 | 0,919 | **0,825** | −0,094 |
| precisão / recall | 0,94 / 0,89 | 0,82 / 0,82 | |

**O vazamento valia 0,08–0,12 de AP.** Na faixa estrita de IoU a queda é de 20%
relativos — coerente com um modelo que reproduzia caixas decoradas, não apenas
reconhecia que havia um carro.

### A resposta

**AP ≈ 0,85 em carro, honesto, numa cena com árvores na frente.** Contra 0,02 do
modelo genérico. Treinar na própria cena recupera, sim, o que a cena adversa tirou
— e o número sobrevive à remoção do vazamento, que era a dúvida real. Note que
`AP ≈ 0,85 em carro` e `mAP 0,660` nem medem a mesma coisa.

### O que estes dados não permitem afirmar

- **Ônibus.** Com 2 ônibus, não existe corte que treine e valide a classe
  honestamente: ou o ônibus fica todo no treino, ou aparece nos dois lados. O bloco
  de validação temporal não tem nenhum. O `AP = 1,000` publicado antes era um ônibus
  parado sendo reencontrado, e sai como resultado.
- **Moto.** Uma instância no dataset inteiro. Não sustenta treino nem avaliação.
  Isso não é limitação do split, é limitação dos dados: duas das três classes do
  `data.yaml` não têm exemplos suficientes para existirem.
- **Generalização para outra cena.** O corte temporal remove o vazamento entre
  treino e validação, mas tudo continua vindo de uma gravação, uma câmera, um
  ângulo. O passo que responderia isso é anotar frames da segunda gravação em
  `data/raw/`, que o modelo nunca viu.
- **A validação limpa é pequena:** 9 frames, 40 caixas, ≈ 17 carros distintos. O
  −0,12 tem margem larga. O que se sustenta é o qualitativo — *o vazamento inflava
  de forma mensurável* — não o valor exato.
- **O split temporal tem um viés próprio:** o fim do vídeo pode ter iluminação ou
  volume de tráfego diferentes do começo (o bloco de validação tem 40 caixas de
  carro em 9 frames, densidade maior que a média do treino). É um preço menor que o
  vazamento, mas existe.
- **O mesmo conjunto de validação escolheu o melhor checkpoint e produziu o
  relatório final.** O ideal seria um terceiro conjunto de teste, nunca visto. Com
  60 imagens não dava para fatiar em três — é limitação de escala, mas empurra o
  resultado para cima de novo.
- **mAP@0,5 é o critério mais generoso.** Vale repetir com `--iou 0.75` para ver se
  as caixas estão realmente justas ou só no lugar certo com o tamanho errado.
- **Anotação humana também erra.** Em veículo muito ocluído, a minha incerteza entra
  no padrão-ouro. Anotar com critério consistente é parte da medição.

### Sobre a contagem

A contagem publicada (125 veículos: 58 às 13:00 e 67 às 16:30; 101 carros, 13
motos, 8 caminhões, 3 ônibus) foi produzida pelo **`yolov8n` genérico**, que é o
padrão de `contagem_carros.py` — ou seja, pelo detector cujo F1 medido nesta cena
é 0,065. Esses números devem ser lidos como demonstração do pipeline de contagem,
não como medição confiável de tráfego.

### As três lições

1. Frames vizinhos de um mesmo vídeo não são exemplos independentes. Contar caixas
   anotadas superestima a amostra.
2. Dividir treino e validação por frame sorteado, em dado de vídeo, é vazamento
   quase garantido. A divisão tem que ser por tempo ou por gravação.
3. Um resultado bom demais para o tamanho do dataset é motivo para investigar, não
   para comemorar. Foi desconfiar do 1,000 que produziu a parte útil deste trabalho.

## Uso de recursos de hardware: velocidade x acurácia x tamanho

Treinar o modelo responde "ele acerta?". Colocá-lo para rodar responde outra
pergunta: **quanto custa cada acerto?** O modelo treinado foi levado pela cadeia
de formatos usada normalmente para produção — PyTorch FP32 → ONNX FP32 →
TensorRT FP16 → TensorRT INT8 — medindo os três eixos em cada etapa.

RTX 3050 Laptop (4 GB), driver 535.230.02, TensorRT 11.2.1.2, torch 2.6.0+cu124,
imgsz 640, batch 1, 200 repetições, IoU 0,5, medido no **modelo do split temporal**
e nos 9 frames de validação limpa:

| formato       | tamanho (MB) | inferência (ms) | FPS   | pesos (MB) | mAP@0,5 | melhor F1 | speedup |
|---------------|--------------|-----------------|-------|------------|---------|-----------|---------|
| PyTorch FP32  | 5,96         | 2,61            | 383,2 | 12,05      | 0,853   | 0,825     | 1,00    |
| ONNX FP32     | 11,70        | 5,05            | 198,1 | 12,05      | 0,870   | 0,815     | 0,52    |
| TensorRT FP16 | 48,07        | **1,43**        | 701,6 | 6,02       | 0,858   | 0,805     | **1,83**|
| TensorRT INT8 | 52,58        | 1,51            | 664,0 | 3,01       | 0,849   | 0,783     | 1,73    |

Controle: o `mAP@0,5 = 0,853` do PyTorch bate exatamente com o AP em carro medido
pela métrica do projeto na seção anterior.

O mesmo benchmark rodado no modelo do split aleatório e no `yolov8n` genérico dá o
mesmo eixo de velocidade (0,52× / 1,85× / 1,75× e 0,52× / 1,80× / 1,77×). Três
modelos diferentes, mesmo padrão: o comportamento é do formato, não do modelo.

Os exports ficam em cache ao lado do `.pt` (`--refazer-export` força de novo). Os
engines do TensorRT levam ~2,5 min cada para construir e **não são portáveis**:
valem para a GPU, o driver e a versão de TensorRT onde foram gerados.

### Três coisas que eu não esperava

São o motivo de valer a pena medir em vez de repetir o que se lê por aí:

- **O ONNX é mais lento que o PyTorch**, quase pela metade, nos três modelos. Ele
  está mesmo na GPU (confirmei o `CUDAExecutionProvider`); o ganho do ONNX Runtime
  viria do `TensorrtExecutionProvider`, e aí a comparação seria com o TensorRT, não
  com o ONNX puro.
- **O INT8 não ganha do FP16** — sai um pouco mais lento e com arquivo maior. A
  primeira suspeita seria quantização que não pegou, mas o grafo desmente: são 492
  tensores de peso em INT8 e 246 pares Q/DQ cobrindo as 64 convoluções. O problema é
  que esses 246 pares de nós custam tempo, e um modelo de 3 M de parâmetros a
  640×640 nesta GPU é limitado por **lançamento de kernel**, não por vazão
  aritmética. Não há cálculo suficiente para os tensor cores INT8 amortizarem o
  overhead que eles mesmos introduzem — e os nós Q/DQ extras são também o motivo de
  o engine INT8 ser *maior* que o FP16. INT8 compensa em modelos maiores, batches
  maiores, ou hardware com grande diferença de vazão entre INT8 e FP16. Nenhuma das
  três condições vale aqui.
- **Quantizar não deixou o arquivo menor.** Comparar o `.engine` de 48 MB com o
  `.pt` de 5,96 MB sugere que o modelo ficou oito vezes maior, o que é falso: os
  dois arquivos não guardam a mesma coisa. O `.pt` guarda pesos; o `.engine` do
  TensorRT 11 guarda pesos *mais* o código de kernel compilado e o plano de
  execução — o próprio log do build entrega o jogo ao reportar
  `Total Weights Memory: 51 MB`, que daria 17 bytes por parâmetro, impossível para
  pesos de verdade. Os 3.011.433 parâmetros dão 12 MB em FP32, 6 MB em FP16 e 3 MB
  em INT8, e os ONNX intermediários confirmam na balança (12,27 / 6,21 / 6,50 MB).
  Por isso a tabela traz duas colunas: `pesos (MB)`, que é o que a quantização de
  fato encolhe, e `tamanho (MB)`, que é o arquivo que se precisa distribuir. Medir
  memória de GPU também não separa as duas coisas — FP16 e INT8 ocupam os mesmos
  178 MiB, porque o contexto CUDA domina o consumo num modelo deste tamanho.
  **Num modelo de 3 M de parâmetros, quantizar não reduz o que você distribui nem o
  que você aloca na GPU.** Quem quer artefato menor está olhando para a alavanca
  errada.

O ponto ótimo é o **TensorRT FP16**: 1,83× mais rápido que o PyTorch sem perda de
acurácia mensurável.

### As quatro decisões de medição que mudam o resultado

Um benchmark de quantização é fácil de fazer de um jeito que não mede nada. Estas
escolhas foram deliberadas.

1. **Latência de inferência pura, separada do fim-a-fim.** Pré e pós-processamento
   (carregar, redimensionar, NMS, montar as caixas) rodam em CPU e são idênticos nos
   quatro formatos. Se o número reportado incluir isso, o ganho da quantização
   aparece diluído por um custo fixo que ela nunca ia tocar. O `speedup` é calculado
   sobre a inferência pura.
2. **A latência é medida sobre uma imagem já em memória.** Leitura de disco e
   decodificação de JPEG variam com o cache do sistema operacional e não têm nada a
   ver com o formato do modelo. São 30 execuções de aquecimento descartadas (a
   primeira inferência sempre paga alocação e autotuning) e 200 medidas, com
   `torch.cuda.synchronize()` a cada uma — sem isso o relógio para antes da GPU
   terminar e todo formato parece infinitamente rápido.
3. **A calibração INT8 usa o split de treino, não o de validação.** O Ultralytics
   calibra no split `val` por padrão (`exporter.py`: `data[self.args.split or "val"]`).
   Como a acurácia também é medida em `val`, o default faria o motor INT8 observar as
   imagens de avaliação durante a calibração. É vazamento mais fraco que treinar em
   `val` — a calibração lê faixas de ativação, não rótulos — mas é vazamento, e daria
   à linha INT8 uma vantagem indevida. `benchmark_formatos.py` força `split="train"`.
4. **A acurácia usa o `nucleo_avaliacao.py` do projeto, não a métrica interna do
   YOLO.** Assim o número de cada formato é comparável com o resto da documentação.
   Usar o `model.val()` do Ultralytics seria comparar duas implementações de mAP
   diferentes.

### Cuidado ao ler a coluna de acurácia

Com 9 frames e 40 caixas, os deltas de mAP (de −0,004 a +0,017) são ruído. O ONNX
FP32 aparece *acima* do PyTorch, o que é impossível — os dois são numericamente
quase idênticos.

O `melhor F1` deixa isso explícito ao inverter de sinal conforme o conjunto: no
modelo vazado ele *subia* nos formatos exportados (0,921 → 0,947), no modelo limpo
ele *cai* (0,825 → 0,783). Uma métrica que troca de direção ao trocar o conjunto
está medindo o conjunto. F1 depende de escolher um limiar numa grade de 20 pontos
sobre 40 caixas — instável por construção. O `mAP`, que não depende de limiar, é o
número em que confiar para comparar formatos.

**O que se sustenta:** nenhum formato degradou de forma detectável, e o FP16
entrega 1,8× de graça. **O que não se sustenta:** ler os deltas de acurácia
individualmente.

Uma ressalva a mais sobre o INT8: o próprio TensorRT avisa durante o build que **a
calibração usou 12 imagens e o recomendado são mais de 300**. Para levar INT8 a
sério aqui, calibrar com um conjunto maior seria o primeiro passo. E os números de
latência valem para esta GPU: uma RTX 3050 Laptop tem pouco poder de cálculo em
relação ao overhead fixo, que é exatamente o motivo de o INT8 não render. Em uma
GPU maior a ordem das linhas pode mudar.

## Limitações conhecidas do código

Estão registradas aqui porque afetam como ler os resultados acima.

- **O contador não consegue usar o modelo treinado.** `contagem_carros.py` fixa
  `VEICULOS = {2: "carro", 3: "moto", 5: "onibus", 7: "caminhao"}` — IDs do COCO — e
  passa `classes=list(VEICULOS)` ao `track`. O modelo treinado neste dataset tem
  `nc: 3`, com IDs 0, 1 e 2, e não existe flag `--classes` neste script (só em
  `avaliar_detector.py`). Rodar o contador com o modelo treinado filtra por classes
  que não existem nele. É por isso que os 125 veículos publicados vieram do
  `yolov8n` genérico. Recontar com o modelo treinado e comparar é o resultado mais
  interessante que o projeto ainda não tem.
- **A avaliação não calibra o contador automaticamente.** O melhor limiar é impresso
  por `avaliar_detector.py`, mas `contagem_carros.py` mantém `--conf 0.35` fixo como
  default. Os ótimos medidos foram 0,05 (genérico) e 0,40 / 0,50 (treinado) — nenhum
  é 0,35. Hoje a ligação é manual.
- **O melhor F1 do genérico está na borda da grade.** A varredura vai de 0,05 a
  0,95 e o ótimo do modelo genérico cai em 0,05, o primeiro ponto, enquanto a
  inferência roda com `--conf-min 0.001`. O ótimo real pode estar abaixo da grade.
- **Label ausente é indistinguível de frame sem veículo.** `ler_gt_yolo` devolve
  caixas vazias em silêncio quando o `.txt` não existe; um erro de caminho derruba a
  métrica sem explicação.
- **`contagem_carros.py` abre o CSV em modo append**, sem verificar janela já
  processada: rodar duas vezes a mesma gravação duplica as contagens.
- **`analise_trafego.py` aplica um único `--minutos` a todas as janelas.** Com
  gravações de durações diferentes, a normalização "veículos/minuto" fica errada sem
  aviso. E com apenas duas janelas medidas, "melhor" e "pior horário" são o máximo e
  o mínimo de dois pontos — a leitura de pico/vale não se sustenta com essa amostra.
- **A linha de contagem é fixa em `0.78`**, sem flag: trocar de cena exige editar o
  código. O sentido é decidido pelo lado **novo** da linha, e não pelo anterior como
  a descrição sugere (os rótulos saem coerentes entre si, a descrição é que está
  trocada). Cruzamento exatamente sobre a linha (`lado == 0`) é perdido.
- **Erro de digitação propagado no dataset:** a classe é `motocycle` no `data.yaml`,
  não `motorcycle`. Está gravado nos pesos treinados e nos caches.

## Ferramentas usadas

- Python
- Ultralytics YOLOv8 para a detecção e para o treino do modelo próprio
- ByteTrack para acompanhar cada veículo entre os quadros
- OpenCV para ler o vídeo e salvar os frames
- NumPy para as contas da avaliação
- pandas para organizar os dados de contagem
- matplotlib para os gráficos
- Roboflow para marcar as caixas à mão
- ONNX Runtime e TensorRT para o benchmark de formatos de deploy

Assista o vídeo no youtube: https://youtube.com/shorts/Du08CHuX0_A
