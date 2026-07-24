# Contagem de Carros

Projeto que conta os veículos que passam em uma rua a partir de vídeos gravados
com a câmera parada. Além de contar, o projeto registra o horário e o sentido de
cada veículo, compara o movimento entre diferentes horários do dia e mede o
quanto o modelo de detecção realmente acerta.

O vídeo usado foi gravado de propósito em um ângulo ruim, com obstáculos na
frente da rua, para testar até onde a técnica funciona em uma situação pouco
favorável.

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

## Como funciona

A detecção usa o YOLOv8, que aponta onde estão os veículos em cada quadro do
vídeo. Como o YOLO não sabe que o carro do quadro 10 é o mesmo do quadro 11, o
ByteTrack entra em seguida e dá um identificador para cada veículo, mantendo esse
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

## Como usar

```bash
pip install -r requirements.txt

# contar os veículos de um vídeo, informando o horário da gravação
python src/contagem_carros.py --video data/raw/video.mp4 --janela "13:00"

# comparar os horários já contados
python src/analise_trafego.py --csv outputs/contagens.csv --minutos 10

# separar frames para marcar à mão
python src/extrair_frames.py --video data/raw/video.mp4 --n 60 --saida data/frames_yolo

# avaliar o modelo genérico
python src/avaliar_detector.py --imagens data/frames_yolo --labels data/labels --modelo yolov8n.pt

# avaliar o modelo treinado neste dataset
python src/avaliar_detector.py --imagens data/frames_yolo --labels data/labels \
    --modelo runs/detect/runs/treino_trafego/weights/best.pt --classes custom
```

## O dataset

O dataset foi feito do zero a partir do próprio vídeo. Foram separados 60 frames
espalhados pela gravação e cada veículo foi marcado à mão, com uma caixa por
veículo, no formato do YOLO (um arquivo de texto por imagem, com a classe e a
posição da caixa em valores de 0 a 1).

São 104 caixas marcadas ao todo, divididas em três classes: 95 carros, 8 ônibus e
1 moto. Para o treino, as imagens foram separadas em 48 para treinar e 12 para
validar.

Essa distribuição é bem desigual, e isso limita o que dá para afirmar: o
resultado de carro tem base razoável, o de ônibus é apertado e o de moto não
permite nenhuma conclusão.

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

Contagem em duas gravações da mesma rua: 125 veículos, sendo 58 na janela das
13:00 e 67 na das 16:30. Do total, 101 carros, 13 motos, 8 caminhões e 3 ônibus.

Na avaliação do detector, a diferença entre usar o modelo pronto e treinar um
modelo com as imagens do próprio vídeo foi grande:

|  | YOLOv8n genérico | YOLOv8n treinado com o vídeo |
|---|---|---|
| mAP com sobreposição de 0,5 | 0,060 | 0,660 |
| Melhor F1 | 0,154 | 0,921 |

O modelo genérico nunca tinha visto esse ângulo, essa distância nem essa
iluminação, e por isso errava ou simplesmente não via boa parte dos veículos.
Treinar com cerca de 50 imagens do próprio cenário resolveu isso. O gargalo não
era a técnica, era a diferença entre o cenário do treino original e o cenário
real.

Vale ler esse número com cuidado:

- 60 frames marcados à mão é pouco. Serve para mostrar que o caminho funciona,
  não para cravar um valor definitivo.
- O resultado zerado da classe moto não quer dizer que o modelo é ruim com motos.
  Quer dizer que havia um único exemplo, o que não permite medir nada.
- O mesmo conjunto de validação foi usado para escolher o melhor ponto do treino
  e para o resultado final. O ideal seria um terceiro conjunto separado, o que
  não cabia em 60 imagens. Por causa disso, o resultado está provavelmente um
  pouco melhor do que seria em imagens novas.
- É provável que o modelo tenha aprendido esse cenário específico, e não trânsito
  em geral. Todas as imagens vêm da mesma câmera e do mesmo ângulo. Para uma
  câmera fixa isso é aceitável, mas não dá para dizer que ele funcionaria em
  outra rua.
- A sobreposição de 0,5 é o critério mais generoso. Repetir a avaliação com 0,75
  mostraria se as caixas estão realmente bem ajustadas.

Os gráficos ficam em `outputs/` e os resultados do treino em
`runs/detect/runs/treino_trafego/`.

## Ferramentas usadas

- Python
- Ultralytics YOLOv8 para a detecção e para o treino do modelo próprio
- ByteTrack para acompanhar cada veículo entre os quadros
- OpenCV para ler o vídeo e salvar os frames
- NumPy para as contas da avaliação
- pandas para organizar os dados de contagem
- matplotlib para os gráficos
- Roboflow para marcar as caixas à mão

Assista o vídeo no youtube: https://youtube.com/shorts/Du08CHuX0_A