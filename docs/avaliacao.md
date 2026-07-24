# Um detector que parece ótimo nas métricas, e por que ele não é

Estudo de caso sobre avaliação de detecção de veículos em vídeo. Começa como uma medição comum de precisão, recall, AP e mAP no YOLOv8, e termina mostrando como um modelo treinado com pouco dado e dado repetido chega a números excelentes sem ter aprendido quase nada.

O ângulo do vídeo, com obstáculos na frente da rua, foi proposital para medir a capacidade da tecnologia em condição ruim. O tamanho pequeno do dataset não foi proposital, mas acabou sendo a parte mais instrutiva: montar o ambiente perfeito de pesquisa nem sempre é a melhor opção, e mostrar os erros que aparecem num modelo mal treinado ensina mais do que exibir um resultado limpo.

---

## Visão geral do fluxo

```
vídeo  ──►  extrair_frames.py  ──►  frames/*.jpg
                                         │
                             anotação manual (ferramenta externa)
                                         │
                                    labels/*.txt  (formato YOLO)
                                         │
frames + labels + modelo  ──►  avaliar_detector.py  ──►  AP por classe, mAP,
                                         │                curva P-R, melhor limiar
                                    nucleo_avaliacao.py  (a matemática, testável)
```

Quatro arquivos: `extrair_frames.py` (amostra frames), `nucleo_avaliacao.py` (a matemática pura, com autoteste), `avaliar_detector.py` (junta YOLO + núcleo + gráficos) e este README.

---

## Passo 1 — Extrair frames para anotar

```bash
python extrair_frames.py --video rua_18h.mp4 --n 120 --saida frames
```

Amostra os frames uniformemente ao longo do vídeo. Eu comecei acreditando que **60 a 100 frames** já davam uma estimativa estável. O experimento mostrou que essa conta está errada, e o motivo é o que este documento explica mais adiante: o que importa não é a quantidade de frames, é o **espaçamento entre eles** e a **variedade de gravações**.

Meu vídeo tem cerca de 310 segundos e eu tirei 60 frames, ou seja, um a cada 5 segundos. Um ônibus leva mais que isso para atravessar a cena, então o mesmo ônibus aparece em vários frames e vira várias caixas anotadas. A amostra parece maior do que é.

A regra melhor: espaçamento maior que o tempo que um veículo leva para cruzar o quadro (uns 15 a 20 segundos aqui), e frames vindos de várias gravações em vez de uma só. Vinte frames de cinco vídeos diferentes valem mais que cem frames de um vídeo. Sortear os frames ao acaso em vez de uniformemente não resolve nada, porque o espaçamento médio continua o mesmo e o sorteio ainda pode juntar dois frames separados por meio segundo. Se você quer estressar o ângulo difícil, use `--inicio` e `--fim` para concentrar a amostragem no trecho com oclusão.

## Passo 2 — Anotar (o trabalho manual, inevitável e valioso)

Isto é o *ground truth* (verdade de referência): você desenha, à mão, a caixa correta de cada veículo em cada frame. É trabalhoso de propósito — é o padrão-ouro contra o qual o modelo é julgado.

Ferramentas gratuitas que exportam no formato YOLO: **LabelImg**, **Label Studio**, **CVAT** ou **Roboflow**. Configure a saída para "YOLO".

**Formato YOLO** (um `.txt` por imagem, mesmo nome do `.jpg`): cada linha é
```
classe  cx  cy  w  h
```
com `cx, cy, w, h` **normalizados** entre 0 e 1 (fração da largura/altura).

Sobre os IDs de classe, o script já resolve os dois casos e você não precisa reanotar nada:

- Anotação exportada pelo Roboflow com as classes do `data.yaml` (`0=ônibus, 1=carro, 2=moto`): é o que eu uso. Para avaliar o modelo genérico, rode com `--classes coco` (o padrão) e o script traduz esses IDs para os do COCO. Para avaliar um modelo treinado neste dataset, rode com `--classes custom`, porque aí o modelo já devolve os mesmos IDs da anotação e nenhuma tradução deve acontecer.
- Anotação feita direto nos IDs do COCO (`2=carro, 3=moto, 5=ônibus, 7=caminhão`): use `--classes coco` e ajuste o dicionário `GT_IDENTIDADE`/`ROBOFLOW_PARA_COCO` no `avaliar_detector.py`.

Trocar essa opção por engano zera o resultado, porque a comparação passa a ser feita entre classes diferentes.

## Passo 3 — Avaliar

```bash
# modelo genérico, pré-treinado no COCO
python src/avaliar_detector.py --imagens data/frames_yolo --labels data/labels \
    --modelo yolov8n.pt --classes coco --iou 0.5

# modelo treinado neste dataset
python src/avaliar_detector.py --imagens data/frames_yolo --labels data/labels \
    --modelo runs/detect/runs/treino_trafego/weights/best.pt --classes custom --iou 0.5
```

Saída: AP por classe, mAP@0.5, o gráfico com as duas curvas e uma tabela `metricas_por_limiar.csv` com precisão/recall/F1 para cada limiar de confiança. Cada execução sobrescreve esses arquivos, então renomeie entre uma e outra se quiser guardar os dois resultados.

---

## A matemática, em ordem de aparição

### IoU — a régua de "acertou a caixa?"

Toda avaliação de detecção começa perguntando: a caixa prevista bate na caixa real? A resposta não é sim/não, é uma medida contínua de sobreposição, o **Intersection over Union**:

$$
\text{IoU}(A, B) = \frac{\text{área}(A \cap B)}{\text{área}(A \cup B)}
$$

0 = não se tocam; 1 = idênticas. Convencionamos um limiar (aqui **0,5**): uma detecção só "acerta" um veículo real se o IoU entre elas for ≥ 0,5. No código, `iou_uma_para_muitas` calcula o IoU de uma predição contra todas as caixas reais do frame de uma vez (vetorizado com numpy).

### Casamento predição↔realidade (o passo delicado)

Detecção não vem rotulada como certa ou errada — é preciso **casar** cada predição com uma caixa real. A regra (`casar_por_imagem`), que é o padrão da área:

1. Ordene as predições do frame por confiança, da maior para a menor.
2. Para cada predição, na ordem, pegue a caixa real de maior IoU **ainda não usada**.
3. Se esse IoU ≥ 0,5 → é um **verdadeiro positivo (TP)**, e aquela caixa real fica "consumida" (não pode casar de novo).
4. Senão → **falso positivo (FP)**.

O detalhe crucial é o "ainda não usada": cada veículo real pode ser contado como acerto **uma única vez**. Se o modelo cospe três caixas no mesmo carro, uma é TP e as outras duas são FP — que é o comportamento correto e, de novo, algo que o seu ângulo difícil provoca bastante.

### Precisão e recall — dois erros que competem

Com cada predição marcada como TP ou FP, e sabendo quantos veículos reais existiam (o total de GT), definimos:

$$
\text{Precisão} = \frac{TP}{TP + FP} \qquad \text{Recall} = \frac{TP}{\text{total de GT}}
$$

- **Precisão** responde: das caixas que o modelo desenhou, quantas estavam certas? (Penaliza alarme falso.)
- **Recall** responde: dos veículos que existiam, quantos o modelo encontrou? (Penaliza o que passou batido.)

As duas puxam para lados opostos: baixar o limiar de confiança acha mais veículos (↑recall) mas inventa mais caixas erradas (↓precisão). Subir o limiar faz o contrário.

### A curva precisão-recall e o AP

Como o equilíbrio depende do limiar, não faz sentido reportar um ponto só. Varremos **todos** os limiares de uma vez: ordenamos todas as predições por confiança e acumulamos TP e FP descendo a lista (`curva_precisao_recall`). Cada posição da lista vira um ponto (recall, precisão) — juntos, formam a **curva precisão-recall**.

O **Average Precision (AP)** é a **área sob essa curva**. Um número entre 0 e 1 que resume o desempenho da classe em todos os limiares. Uso a interpolação "all-points" (padrão COCO): antes de integrar, torno a curva monótona pegando, em cada ponto, a maior precisão à direita (`average_precision`). Isso remove o serrilhado e é a definição padrão da literatura.

### mAP — a nota final

O **mean Average Precision** é a média dos APs sobre as classes que têm pelo menos um exemplo real no seu conjunto:

$$
\text{mAP} = \frac{1}{|\text{classes}|}\sum_{c} \text{AP}_c
$$

`mAP@0.5` quer dizer "mAP com o limiar de IoU em 0,5". É a métrica que você reporta como resumo — e a que aparece em papers de detecção.

### Escolha do limiar de operação (o painel da direita)

O AP avalia o modelo em *todos* os limiares, mas para **rodar** o contador você precisa fixar **um**. A tabela por limiar (`metricas_por_limiar`) calcula precisão, recall e F1 para cada valor de confiança, e o script marca o de **maior F1** — a média harmônica que equilibra os dois erros:

$$
F_1 = \frac{2 \cdot \text{Precisão} \cdot \text{Recall}}{\text{Precisão} + \text{Recall}}
$$

Esse é o limiar que você deveria passar ao `contagem_carros.py`. Ou seja: a avaliação não é só nota, é o que **calibra** o seu contador.

---

## Por que confiar neste código

`nucleo_avaliacao.py` roda um autoteste sozinho:

```bash
python nucleo_avaliacao.py
```

Ele confere a matemática do AP em três casos que dá para verificar na mão: detecções perfeitas (AP = 1,000), um falso positivo no meio da lista (AP ≈ 0,833) e nenhuma detecção (AP = 0,000). Se algum falhar, o `assert` quebra. Separei a matemática da inferência justamente para ela ser testável sem depender do modelo — é assim que se garante que o número que você vai publicar está certo.

---
## Experimento: treinei um modelo no meu próprio dataset

Depois de rodar a avaliação com o YOLOv8n genérico (pré-treinado no COCO), resolvi testar quanto eu ganharia especializando o modelo no meu próprio vídeo. Peguei os 60 frames anotados, separei 48 para treino e 12 para validação, e treinei um YOLOv8n do zero (partindo dos pesos COCO) por 60 epochs em CPU.

O resultado:

| | YOLOv8n genérico (COCO) | YOLOv8n treinado no meu dataset |
|---|---|---|
| mAP@0.5 | 0,060 | 0,660 |
| Melhor F1 | 0,154 | 0,921 |

Por classe, o modelo treinado dá AP de 1,000 em ônibus e 0,981 em carro.

A primeira leitura é a óbvia: o modelo genérico nunca tinha visto o ângulo, a distância nem a iluminação do meu vídeo, então errava ou simplesmente não detectava boa parte dos veículos, e especializar no meu domínio resolveu. Essa parte é verdade. Só que AP de 1,000 com 60 frames anotados não é resultado de modelo bom, é sinal de que a avaliação está medindo a coisa errada. Foi investigar isso que virou o assunto principal do estudo.

## O que estava errado (e é o mais interessante do projeto)

### O mesmo veículo foi contado muitas vezes

São 163 caixas anotadas: 134 carros, 28 ônibus e 1 moto. Parece uma amostra pequena mas razoável. Não é.

Os 60 frames saíram de um vídeo de 310 segundos, um frame a cada 5 segundos. Um ônibus demora mais que isso para atravessar a cena, então as 28 caixas de ônibus não são 28 ônibus, são talvez cinco ou seis ônibus fotografados várias vezes seguidas. Os 134 carros também são bem menos que 134 carros distintos.

A conclusão desconfortável: o número de caixas anotadas não mede o tamanho da amostra. O que mede é a quantidade de veículos diferentes, e essa eu não tenho anotada em lugar nenhum.

### Pior: o mesmo ônibus ficou nos dois lados da divisão

Se o mesmo ônibus aparece em vários frames e esses frames foram divididos ao acaso entre treino e validação, então parte dos veículos usados para testar o modelo é a mesma que ele viu treinando, só que meio segundo depois. Ele não está reconhecendo um ônibus, está reconhecendo aquele ônibus.

É isso que explica o AP de 1,000 em ônibus, que nenhum detector honesto tira com esse tamanho de dataset. A nota alta não veio de aprendizado, veio de vazamento entre treino e validação.

A correção é dividir por tempo e não por frame sorteado: os primeiros 80 por cento do vídeo para treino e os últimos 20 por cento para validação, ou melhor ainda, treinar com um vídeo e validar inteiramente com outro. Aí nenhum veículo aparece dos dois lados.

### A nota muda conforme o conjunto onde você mede

Isso dá para demonstrar em vez de só afirmar. As mesmas duas avaliações, rodadas nos 60 frames anotados e depois só nos 12 frames de validação:

| | genérico (60 frames) | genérico (12 de validação) | treinado (60 frames) | treinado (12 de validação) |
|---|---|---|---|---|
| mAP@0,5 | 0,060 | 0,020 | 0,660 | 0,660 |
| Melhor F1 | 0,154 | 0,065 | 0,943 | 0,921 |

O modelo treinado vai melhor nos 60 frames porque 48 deles são exatamente as imagens em que ele treinou. É a nota de uma prova com as respostas anotadas na mesa. O 0,921 da validação é o número menos ruim dos dois, e mesmo ele está contaminado pelo vazamento descrito acima.

### A classe moto nunca foi um resultado

Existe 1 moto no dataset inteiro, e ela caiu na validação. AP de 0,000 nessa classe não quer dizer que o modelo é ruim com motos, quer dizer que um exemplo não permite medir nada. Prefiro tratar como "dados insuficientes" do que publicar o número como se fosse desempenho.

E vale notar o efeito colateral: como o mAP é a média das classes, esse zero sozinho puxa a nota geral de perto de 0,99 para 0,660. O número que eu reportaria como resumo do modelo é, na prática, decidido por uma única moto.

### Outras ressalvas que continuam valendo

- **Usei o mesmo conjunto de validação para escolher o melhor checkpoint e para o relatório final.** O ideal seria um terceiro conjunto de teste, nunca visto no treino. Com 60 imagens não dava para fatiar em três, então é uma limitação de escala, mas ela empurra o resultado para cima de novo.
- **O modelo provavelmente decorou o cenário.** 60 épocas em 48 imagens, mesma câmera, mesmo ângulo, poucos veículos se repetindo. Para uma câmera fixa isso até é o caso de uso real, mas é diferente de dizer que ele funciona em outra rua.
- **mAP@0,5 é o critério mais generoso.** Vale repetir com `--iou 0.75` para ver se as caixas estão realmente justas ou só no lugar certo com o tamanho errado.
- **Anotação humana também erra.** Em veículo muito ocluído, a minha incerteza entra no padrão-ouro. Anotar com critério consistente é parte da medição.

## O que esse experimento ensina

Se eu tivesse que resumir numa frase: treinar com cerca de 50 frames do próprio cenário levou o mAP@0,5 de 0,06 para 0,66 e o melhor F1 de 0,15 para 0,92, mas boa parte desse salto é o modelo reencontrando os mesmos veículos que ele viu treinando, e não aprendizado de verdade.

O ganho de especializar no próprio domínio existe e é grande. O que não dá para fazer é usar esses números como prova dele, porque a forma como o dataset foi montado e dividido garante que eles saiam altos de qualquer jeito.

As três lições que eu levo:

1. Frames vizinhos de um mesmo vídeo não são exemplos independentes. Contar caixas anotadas superestima a amostra.
2. Dividir treino e validação por frame sorteado, em dado de vídeo, é vazamento quase garantido. A divisão tem que ser por tempo ou por gravação.
3. Um resultado bom demais para o tamanho do dataset é motivo para investigar, não para comemorar. Foi desconfiar do 1,000 que produziu a parte útil deste trabalho.

Montar o ambiente perfeito de pesquisa nem sempre é a melhor opção. Mostrar onde um modelo mal treinado engana quem está medindo é mais útil, e é o tipo de erro que aparece bastante em projeto real com câmera fixa.
