# Contagem de Carros

Projeto de análise de tráfego a partir de vídeos de uma rua, usando detecção e
rastreamento de veículos por visão computacional. O objetivo é contar quantos
veículos passam em cada horário do dia, identificar o sentido do deslocamento
(subindo/descendo) e comparar o volume de tráfego entre diferentes janelas de
horário para descobrir qual é o pior (e o melhor) horário na via observada.

## Como funciona

1. Um vídeo fixo é gravado apontando para a rua.
2. `contagem_carros.py` roda detecção e rastreamento de veículos (YOLOv8 +
   ByteTrack) e conta um veículo quando o centro da sua caixa cruza uma linha
   virtual definida no quadro, registrando horário, classe do veículo
   (carro, moto, ônibus, caminhão) e sentido em `contagens.csv`.
3. `analise_trafego.py` lê o CSV consolidado (podendo conter várias janelas de
   horário/vídeos) e gera um gráfico comparando o volume de veículos por
   minuto entre os horários, além da composição da frota por tipo de veículo.
4. `extrair_frames_exemplo.py` / `extrair_frames_13h.py` e
   `extrair_trechos_13h.py` geram, respectivamente, frames PNG e trechos de
   vídeo (crus e com as marcações do algoritmo) para ilustrar o funcionamento
   da detecção.

## Stack

- Python
- [OpenCV](https://opencv.org/) — leitura e manipulação de vídeo/frames
- [Ultralytics YOLOv8](https://docs.ultralytics.com/) — detecção de objetos
- [ByteTrack](https://github.com/ifzhang/ByteTrack) — rastreamento multi-objeto
- [pandas](https://pandas.pydata.org/) — manipulação dos dados de contagem
- [matplotlib](https://matplotlib.org/) — geração dos gráficos
- [NumPy](https://numpy.org/) — suporte numérico
- [FFmpeg](https://ffmpeg.org/) — extração de trechos de vídeo

## Uso

```bash
# Contar veículos em um vídeo, informando o horário de início da gravação
python contagem_carros.py --video video.mp4 --janela "13:00"

# Gerar o gráfico comparativo a partir do CSV consolidado
python analise_trafego.py --csv contagens.csv
```
