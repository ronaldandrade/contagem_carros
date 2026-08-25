# Contagem de Carros

Contagem de veículos em vídeo de câmera fixa, com detecção, rastreamento e
avaliação medida.

## Objetivo

Medir até onde uma solução de visão computacional funciona em condições ruins. O
vídeo foi gravado de propósito em ângulo desfavorável, com árvores entre a câmera
e a rua, deixando os veículos parcialmente ocluídos e pequenos na imagem.

O projeto responde três perguntas:

1. Um detector aguenta essa cena?
2. O número que ele reporta é confiável?
3. Quanto de velocidade dá para extrair do hardware na hora de rodar?

## O que faz

- Detecta e rastreia os veículos quadro a quadro.
- Conta cada veículo ao cruzar uma linha virtual, registrando horário, tipo e
  sentido em CSV.
- Compara o movimento entre horários diferentes de gravação.
- Avalia o detector contra caixas marcadas à mão, com implementação própria de
  IoU, precisão, recall, F1 e AP.
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

Construído do zero a partir do próprio vídeo: 60 frames amostrados ao longo da
gravação, com cada veículo marcado à mão em formato YOLO. São 163 caixas em três
classes — 134 carros, 28 ônibus e 1 moto.

A divisão treino/validação é feita por **bloco de tempo contíguo com zona-tampão**
(treino nos frames 0–47, 48–51 descartados, validação em 52–65), e não por sorteio.
Frames vizinhos de um mesmo vídeo mostram os mesmos veículos: sortear frames coloca
o mesmo carro nos dois lados da divisão e infla o resultado.

Duas classes não são avaliáveis com esses dados: as 28 caixas de ônibus são **2
ônibus** (um parado por 27 frames seguidos) e há **1 única caixa de moto**. Os
resultados abaixo são da classe carro.

## Resultados

**Detecção — classe carro, validação limpa:**

| | AP@0,5 | melhor F1 |
|---|---|---|
| YOLOv8n genérico (COCO) | 0,053 | 0,122 |
| YOLOv8n treinado na cena | **0,853** | 0,825 |

Ambos medidos no mesmo conjunto de validação limpo, com a mesma implementação de AP.

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
python src/extrair_frames.py --video data/raw/video.mp4 --n 60 --saida data/frames

# avaliar o detector
python src/avaliar_detector.py --imagens data/splits/temporal/images/val \
    --labels data/splits/temporal/labels/val --modelo models/split_temporal.pt --classes custom

# montar o split temporal e treinar
python src/preparar_split.py --val-inicio 52 --tampao 4
yolo detect train model=yolov8n.pt data=data/splits/temporal/data.yaml \
    epochs=60 batch=8 imgsz=640 seed=0

# medir velocidade x acurácia x tamanho por formato (exige GPU NVIDIA)
python src/benchmark_formatos.py \
    --modelo models/split_temporal.pt --classes custom
```

## Estrutura

Fonte e derivado são separados: `data/labels/` é o único material insubstituível e
vai versionado; frames, splits, modelos e saídas são reconstruídos por comando.

```
data/raw/         vídeos originais          (ignorado)
data/frames/      60 frames extraídos       (ignorado, regerável)
data/labels/      60 rótulos feitos à mão   VERSIONADO
data/splits/      splits gerados            (ignorado, regerável)
models/           modelos treinados         (ignorado)
outputs/          tabelas e gráficos

src/contagem_carros.py     detecção + rastreamento + contagem
src/analise_trafego.py     comparação entre horários
src/extrair_frames.py      amostragem de frames para anotação
src/nucleo_avaliacao.py    IoU, precisão/recall, AP (numpy puro, com autoteste)
src/deteccao.py            ponte YOLO -> arrays (mantém o núcleo livre de YOLO)
src/avaliar_detector.py    avaliação do detector e gráficos
src/preparar_split.py      gera os splits treino/validação a partir da fonte
src/benchmark_formatos.py  benchmark PyTorch/ONNX/TensorRT
```

Rodar `python src/nucleo_avaliacao.py` executa o autoteste das contas de avaliação
em três casos verificáveis à mão.

## Documentação

- [`docs/estudo_de_caso.md`](docs/estudo_de_caso.md) — o estudo completo: as métricas
  e a matemática da avaliação, o vazamento treino/validação medido e corrigido pelo
  split temporal, o trade-off entre formatos de deploy e as limitações conhecidas.

Vídeo: https://youtube.com/shorts/Du08CHuX0_A
