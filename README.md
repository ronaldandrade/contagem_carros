# Contagem de Carros

Contagem de veículos em vídeo de câmera fixa, com detecção, rastreamento e
avaliação medida.

## Objetivo
Medir até onde uma solução de visão computacional funciona em condições ruins. O
vídeo foi gravado de propósito em ângulo desfavorável, com árvores entre a câmera
e a rua, deixando os veículos parcialmente ocluídos e pequenos na imagem.

Dataset: O quão o tamanho e qualidade do dataset importam. Feito o treinamento com um dataset pequeno de 60 imagens ( com split de 47 para treino, 4 descartados na zona-tampão e 9 para validação ).  Um segundo dataset com 281 imagens ( 228 para treino, 45 para validação ), com split muito mais espaçado.  

O projeto responde três perguntas:

1. Um detector aguenta essa cena?
2. O número que ele reporta é confiável?
3. Quanto de velocidade dá para extrair do hardware na hora de rodar?

## O que faz

- Detecta e rastreia os veículos quadro a quadro.
- Conta cada veículo ao cruzar uma linha virtual, registrando horário, tipo e  sentido em CSV.
- Compara o movimento entre horários diferentes de gravação.
- Avalia o detector contra caixas marcadas à mão, com implementação própria de  IoU, precisão, recall, F1 e AP.
- Treina um modelo com as imagens da própria cena e compara com o modelo genérico.
- Exporta para ONNX e TensorRT (FP16/INT8) e mede velocidade, acurácia e tamanho.

## Tecnologias

| | |
|---|---|
| Detecção | Ultralytics YOLOv8 |
| Rastreamento | ByteTrack |
| Vídeo e imagem | OpenCV |
| Cálculo e dados | NumPy, pandas |
| Gráficos | matplotlib |
| Deploy | ONNX Runtime, TensorRT |
| Anotação | Roboflow |

## Dataset

Construído do zero a partir do próprio vídeo: 281 frames amostrados ao longo da
gravação, com cada veículo marcado à mão em formato YOLO.

A divisão treino/validação é feita por **bloco de tempo contíguo** pois:
frames vizinhos de um mesmo vídeo mostram os mesmos veículos,logo sortear frames coloca
o mesmo carro nos dois lados da divisão e infla o resultado.

## Resultados

**Detecção — classe carro, validação limpa do dataset grande (45 frames, 184 caixas):**

| | AP@0,5 | melhor F1 |
|---|---|---|
| YOLOv8n genérico (COCO) | 0,029 | 0,071 |
| YOLOv8n treinado nas 60 imagens | 0,187 | 0,265 |
| YOLOv8n treinado nas 281 imagens | **0,634** | 0,544 |

Os três medidos no mesmo conjunto de validação limpo, com a mesma implementação de AP.

Quatro vezes mais imagem triplica o AP. Na cena original, isolada, o salto é de
0,298 para 0,590; no vídeo novo, que o dataset pequeno nunca viu, é de 0,076 para
0,697.

**O mesmo dataset grande, medido com split sorteado em vez de temporal:** AP@0,5
de 0,747 contra 0,634. Os 0,11 de diferença são vazamento, não desempenho.

**Detecção no dataset pequeno, medida na validação dele (9 frames, 40 caixas):**

| | AP@0,5 | melhor F1 |
|---|---|---|
| YOLOv8n genérico (COCO) | 0,053 | 0,122 |
| YOLOv8n treinado na cena | **0,853** | 0,825 |

Este 0,853 **não se compara** com o 0,634 acima: são conjuntos de validação
diferentes. A validação pequena tem 4,4 carros por frame e a grande tem 7,3 —
cena bem mais cheia, onde todo modelo pontua mais baixo. Os 60 frames antigos
reaparecem no dataset novo com rótulo idêntico, então a diferença é de cena, não
de reanotação.

Um modelo pronto de prateleira praticamente não funciona nesta cena. Treinar com as
imagens do próprio local recupera o desempenho, e o número sobrevive à remoção do
vazamento treino/validação.

**Deploy — RTX 3050 Laptop, imgsz 640, batch 1:**

| formato | inferência | FPS | tamanho | speedup |
|---|---|---|---|---|
| PyTorch FP32 | 2,61 ms | 383 | 5,96 MB | 1,00 |
| ONNX FP32 | 5,05 ms | 198 | 11,70 MB | 0,52 |
| **TensorRT FP16** | **1,43 ms** | 702 | 48,07 MB | **1,83** |
| TensorRT INT8 | 1,51 ms | 664 | 52,58 MB | 1,73 |

TensorRT FP16 é o ponto ótimo: 1,83× mais rápido sem perda de acurácia detectável.
O ONNX no CUDAExecutionProvider fica mais lento que o PyTorch, e o INT8 não ganha
do FP16 neste tamanho de modelo.

## Como usar

```bash
pip install -r requirements.txt

# contar os veículos de um vídeo
python src/contagem_carros.py --video data/raw/video.mp4 --janela "13:00"

# comparar horários já contados
python src/analise_trafego.py --csv outputs/contagens.csv --minutos 10

# separar frames para anotar à mão
python src/extrair_frames.py --video data/raw/video.mp4 --n 60 --saida data/data2/frames/IMG_9575

# montar o split temporal do dataset grande e treinar
python src/preparar_split_data2.py
yolo detect train model=yolov8n.pt data=data/splits2/temporal/data.yaml \
    epochs=60 batch=8 imgsz=640 seed=0

# avaliar o detector
python src/avaliar_detector.py --imagens data/splits2/temporal/images/val \
    --labels data/splits2/temporal/labels/val --modelo models/data2_temporal.pt --classes custom4

# refazer o dataset pequeno e o braço de controle sorteado, para comparar
python src/preparar_split.py --val-inicio 52 --tampao 4
python src/preparar_split_data2.py --modo roboflow

# medir velocidade x acurácia x tamanho por formato (exige GPU NVIDIA)
python src/benchmark_formatos.py \
    --modelo models/data2_temporal.pt --classes custom4
```

## Estrutura

O repositório versiona **os rótulos e o código**, nada mais: `data/data2/labels/`
são os 281 arquivos marcados à mão, e `data/exemplo/` traz dois pares
imagem+rótulo para o formato ficar legível sem baixar nada. Vídeo, frames e
imagens ficam fora — são centenas de MB e não pertencem a um repositório de
código. A regra do `.gitignore` é por extensão, e não por pasta, para continuar
valendo se o dataset for reorganizado.

Os 60 rótulos do dataset pequeno não têm pasta própria: eles reaparecem
inalterados dentro do export do grande, com o mesmo nome de arquivo, e é de lá
que o split pequeno é remontado.

**As imagens não são regeráveis a partir do vídeo.** `src/extrair_frames.py`
amostra por tempo, e os parâmetros usados para montar `data/data2/frames/` não
ficaram registrados: reextrair com os valores óbvios cai em outros instantes do
vídeo (diferença média de 50/255 por pixel), e aí os rótulos não alinham mais.
Para reproduzir o treino é preciso o export do Roboflow, não só este repositório.
Os splits, esses sim, saem por comando a partir das imagens e dos rótulos.

```
data/raw/            vídeos originais              (ignorado)
data/data1/frames/   60 frames do dataset pequeno  (ignorado)
data/data2/frames/   frames extraídos dos 2 vídeos (ignorado)
data/data2/image/    281 imagens do export         (ignorado)
data/data2/labels/   281 rótulos feitos à mão      VERSIONADO
data/exemplo/        2 pares imagem+rótulo         VERSIONADO
data/splits1/        split do dataset pequeno      (ignorado, regerável)
data/splits2/        splits do dataset grande      (ignorado, regerável)
models/              modelos treinados             (ignorado)
outputs/             tabelas e gráficos

src/contagem_carros.py       detecção + rastreamento + contagem
src/analise_trafego.py       comparação entre horários
src/extrair_frames.py        amostragem de frames para anotação
src/nucleo_avaliacao.py      IoU, precisão/recall, AP (numpy puro, com autoteste)
src/deteccao.py              ponte YOLO -> arrays (mantém o núcleo livre de YOLO)
src/avaliar_detector.py      avaliação do detector e gráficos
src/preparar_split.py        split do dataset pequeno (60 imagens)
src/preparar_split_data2.py  splits do dataset grande (281 imagens, 2 vídeos)
src/benchmark_formatos.py    benchmark PyTorch/ONNX/TensorRT
```

Rodar `python src/nucleo_avaliacao.py` executa o autoteste das contas de avaliação
em três casos verificáveis à mão.

## Documentação

- [`docs/estudo_de_caso.md`](docs/estudo_de_caso.md) — o estudo completo: as métricas
  e a matemática da avaliação, o vazamento treino/validação medido e corrigido pelo
  split temporal, o trade-off entre formatos de deploy e as limitações conhecidas.

Vídeo: https://youtube.com/shorts/Du08CHuX0_A
