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

# avaliar o modelo genérico
python src/avaliar_detector.py --imagens data/splits/temporal/images/val \
    --labels data/splits/temporal/labels/val --modelo models/yolov8n.pt --classes coco

# avaliar o modelo treinado neste dataset
python src/avaliar_detector.py --imagens data/splits/temporal/images/val \
    --labels data/splits/temporal/labels/val --modelo models/treino_original.pt --classes custom
```

## O dataset

O dataset foi feito do zero a partir do próprio vídeo. Foram separados 60 frames
espalhados pela gravação e cada veículo foi marcado à mão, com uma caixa por
veículo, no formato do YOLO (um arquivo de texto por imagem, com a classe e a
posição da caixa em valores de 0 a 1).

São 163 caixas marcadas ao todo, divididas em três classes: carros, ônibus e  moto. Para o treino, as imagens foram separadas em 48 para treinar e 12 para
validar — dataset extremamente pequeno, mas proposital, a principio, para evoluções no estudos mais a frente.


## As medidas usadas

Antes de tudo é preciso decidir quando uma caixa prevista conta como acerto. A
regra é a sobreposição entre a caixa prevista e a caixa marcada à mão, dividida
pela área total que as duas ocupam juntas. Vale 0 quando não se tocam e 1 quando
são iguais. O padrão adotado aqui é considerar acerto a partir de 0,5.

Cada previsão é então comparada com as caixas reais do frame, das mais confiantes
para as menos confiantes. Cada veículo real só pode ser acertado uma vez: se o
modelo desenha três caixas no mesmo carro, uma conta como acerto e as outras duas
como erro.

Com isso, saem as medidas:

- Precisão: das caixas que o modelo desenhou, quantas estavam certas.
- Recall: dos veículos que existiam, quantos o modelo encontrou.
- F1: um número único que equilibra os dois anteriores.
- AP: resume o desempenho de uma classe considerando todos os níveis de
  confiança de uma vez, em vez de escolher um.
- mAP: a média do AP entre as classes que têm exemplos suficientes.

Precisão e recall puxam para lados opostos. Exigir mais confiança do modelo faz
ele errar menos, mas deixar mais veículos passarem. Exigir menos faz o contrário.
Por isso o projeto gera uma tabela com essas medidas em cada nível de confiança e
aponta qual dá o melhor equilíbrio. Esse valor é o que deve ser usado na
contagem, ou seja, a avaliação não serve só para dar uma nota, ela também ajusta
o contador.

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
já tinha visto.

### O diagnóstico, em números

Três medições feitas nos próprios rótulos:

- **58% das caixas têm IoU ≥ 0,5 com uma caixa da mesma classe no frame anterior**
  (40% com IoU ≥ 0,7). É o mesmo veículo, quase na mesma posição.
- **Os 12 frames de validação tinham distância 1 para o treino. Todos os doze.**
  Não era um split parcialmente contaminado — não sobrava um frame limpo sequer.
- **As 28 caixas de "ônibus" são 2 ônibus:** um parado nos frames 7 a 34
  (27 caixas), mais um no frame 47. E há **1 única caixa de moto** no dataset.

Isso desmonta o `0,660`: ele era a média de `ônibus 1,000` (um ônibus decorado),
`carro 0,974` (inflado) e `moto 0,000` (uma instância nunca detectada). Nenhuma das
três parcelas media generalização.

Importante: **os rótulos estavam certos.** Um ônibus marcado em 27 frames são 27
caixas corretas. O erro não foi de anotação, foi de *amostragem e divisão* — e por
isso a correção não exigiu re-anotar nada.

### Segundo treino: split temporal com zona-tampão

`src/preparar_split.py` divide por bloco de tempo contíguo, descartando uma faixa
de frames na fronteira para que nenhum veículo apareça dos dois lados:

```
frames 0–47   |   48–51    |   52–65
  TREINO      |   TAMPÃO   |  VALIDAÇÃO
 47 frames    | descartado |  9 frames
```

A distância mínima entre validação e treino sobe de **1 para 5 frames** (~35 s).

Para que a diferença fosse atribuível ao split e não ao treino, **os dois braços
foram retreinados do zero com hiperparâmetros idênticos** (yolov8n, 60 épocas,
batch 8, imgsz 640, seed 0). Controle de sanidade: o braço aleatório retreinado
reproduz o modelo histórico (0,652 contra 0,660).

Comparação honesta só na classe **carro** — a única com exemplos dos dois lados
do corte:

| classe carro | split aleatório (vazado) | split temporal (limpo) | queda |
|---|---|---|---|
| AP@0,5 | 0,974 | **0,853** | −0,121 |
| AP@0,5:0,95 | 0,502 | **0,402** | −0,100 |
| melhor F1 | 0,919 | **0,825** | −0,094 |
| precisão / recall | 0,94 / 0,89 | 0,82 / 0,82 | |

**O vazamento valia ~0,12 de AP.** Na faixa estrita de IoU a queda é de 20%
relativos — coerente com um modelo que reproduzia caixas decoradas, não apenas
reconhecia que havia um carro.

### A resposta

**AP ≈ 0,85 em carro, honesto, numa cena com árvores na frente.** Contra 0,02 do
modelo genérico. Treinar na própria cena recupera, sim, o que a cena adversa tirou
— e o número sobrevive à remoção do vazamento, que era a dúvida real.

### O que estes dados não permitem afirmar

- **Ônibus.** Com 2 ônibus, não existe corte que treine e valide a classe
  honestamente. O `AP = 1,000` publicado antes era um ônibus parado sendo
  reencontrado, e sai como resultado.
- **Moto.** Uma instância no dataset inteiro. Não sustenta treino nem avaliação.
- **Generalização para outra cena.** O corte temporal remove o vazamento entre
  treino e validação, mas tudo continua vindo de uma gravação, uma câmera, um
  ângulo. O passo que responderia isso é anotar frames da segunda gravação em
  `data/raw/`, que o modelo nunca viu.
- **A validação limpa é pequena:** 9 frames, 40 caixas, ≈ 17 carros distintos. O
  −0,12 tem margem larga. O que se sustenta é o qualitativo — *o vazamento inflava
  de forma mensurável* — não o valor exato.

### Sobre a contagem

A contagem publicada (125 veículos: 58 às 13:00 e 67 às 16:30; 101 carros, 13
motos, 8 caminhões, 3 ônibus) foi produzida pelo **`yolov8n` genérico**, que é o
padrão de `contagem_carros.py` — ou seja, pelo detector cujo F1 medido nesta cena
é 0,065. Esses números devem ser lidos como demonstração do pipeline de contagem,
não como medição confiável de tráfego.

Os detalhes completos estão em `docs/split_temporal.md` e `docs/avaliacao.md`.

## Uso de recursos de hardware: velocidade x acurácia x tamanho

Treinar o modelo responde "ele acerta?". Colocá-lo para rodar responde outra
pergunta: **quanto custa cada acerto?** O modelo treinado foi levado pela cadeia
de formatos usada normalmente para produção — PyTorch FP32 → ONNX FP32 →
TensorRT FP16 → TensorRT INT8 — medindo os três eixos em cada etapa.

RTX 3050 Laptop, imgsz 640, batch 1, 200 repetições, medido no **modelo do split
temporal** e nos 9 frames de validação limpa:

| formato       | tamanho (MB) | inferência (ms) | FPS   | pesos (MB) | mAP@0,5 | speedup |
|---------------|--------------|-----------------|-------|------------|---------|---------|
| PyTorch FP32  | 5,96         | 2,61            | 383,2 | 12,05      | 0,853   | 1,00    |
| ONNX FP32     | 11,70        | 5,05            | 198,1 | 12,05      | 0,870   | 0,52    |
| TensorRT FP16 | 48,07        | **1,43**        | 701,6 | 6,02       | 0,858   | **1,83**|
| TensorRT INT8 | 52,58        | 1,51            | 664,0 | 3,01       | 0,849   | 1,73    |

O mesmo benchmark rodado no modelo do split aleatório e no `yolov8n` genérico dá o
mesmo eixo de velocidade (0,52× / 1,85× / 1,75× e 0,52× / 1,80× / 1,77×). Três
modelos diferentes, mesmo padrão: o comportamento é do formato, não do modelo.

Três coisas que eu não esperava, e que são o motivo de valer a pena medir em vez
de repetir o que se lê por aí:

- **O ONNX é mais lento que o PyTorch**, quase pela metade. Ele está mesmo na GPU
  (confirmei o `CUDAExecutionProvider`); o ganho do ONNX Runtime viria do
  execution provider de TensorRT, não do ONNX puro.
- **O INT8 não ganha do FP16** — sai um pouco mais lento e com arquivo maior. A
  quantização funcionou (o grafo tem 492 tensores em INT8 e 246 pares Q/DQ nas 64
  convoluções); o problema é que esses pares de nós custam tempo, e um modelo de
  3 M de parâmetros nesta GPU é limitado por lançamento de kernel, não por vazão
  aritmética. Não há cálculo suficiente para os tensor cores INT8 pagarem o
  overhead que eles mesmos introduzem.
- **Quantizar não deixou o arquivo menor.** Comparar o `.engine` de 48 MB com o
  `.pt` de 5,96 MB sugere que o modelo ficou oito vezes maior, o que é falso: os
  dois arquivos não guardam a mesma coisa. O `.pt` guarda pesos; o `.engine` do
  TensorRT 11 guarda pesos mais o código de kernel compilado. Os pesos foram de
  12 MB para 3 MB, como esperado — só que num modelo deste tamanho os pesos já
  eram a menor parte do custo.

O ponto ótimo é o **TensorRT FP16**: 1,83× mais rápido que o PyTorch sem perda de
acurácia mensurável.

Uma ressalva sobre a coluna de acurácia: com 9 frames e 40 caixas, os deltas de mAP
(de −0,004 a +0,017) são ruído. O ONNX FP32 aparece *acima* do PyTorch, o que é
impossível — os dois são numericamente quase idênticos. O que se sustenta é o
negativo: **nenhum formato degradou de forma detectável.** Ler os deltas
individualmente, não.

A discussão completa — incluindo por que a calibração INT8 precisa usar o split de
treino, e por que o `melhor F1` inverte de sinal conforme o conjunto — está em
`docs/benchmark_formatos.md`.

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