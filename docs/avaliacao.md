# Avaliação do detector YOLOv8 — precisão-recall, AP e mAP

Avaliação de desempenho do YOLO em vídeo de fluxo de carros. O ângulo do video com obstáculos a frente foi proposital para medir a capacidade da técnologia.
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

Amostra 60 frames uniformemente ao longo do vídeo. Para um estudo honesto, **60-100 frames** já dão uma estimativa estável sem tornar a anotação interminável. Se você quer estressar o ângulo difícil, use `--inicio` e `--fim` para concentrar a amostragem no trecho com oclusão.

## Passo 2 — Anotar (o trabalho manual, inevitável e valioso)

Isto é o *ground truth* (verdade de referência): você desenha, à mão, a caixa correta de cada veículo em cada frame. É trabalhoso de propósito — é o padrão-ouro contra o qual o modelo é julgado.

Ferramentas gratuitas que exportam no formato YOLO: **LabelImg**, **Label Studio**, **CVAT** ou **Roboflow**. Configure a saída para "YOLO".

**Formato YOLO** (um `.txt` por imagem, mesmo nome do `.jpg`): cada linha é
```
classe  cx  cy  w  h
```
com `cx, cy, w, h` **normalizados** entre 0 e 1 (fração da largura/altura). Use os mesmos IDs de classe do COCO que o pipeline usa: `2=carro, 3=moto, 5=ônibus, 7=caminhão`. (Se a sua ferramenta gerar IDs 0,1,2,3, ajuste o dicionário `CLASSES` nos scripts.)

## Passo 3 — Avaliar

```bash
python avaliar_detector.py --imagens frames --labels labels --modelo yolov8n.pt --iou 0.5
```

Saída: AP por classe, mAP@0.5, o gráfico com as duas curvas e uma tabela `metricas_por_limiar.csv` com precisão/recall/F1 para cada limiar de confiança.

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

## Como isso vira munição

- **Post de LinkedIn (o do rigor):** "Não basta rodar o YOLO — tem que avaliar." O gráfico da curva P-R + a frase sobre escolher o limiar por F1. É o post que sinaliza maturidade, não entusiasmo.
- **Corte curto (60s):** "O que é precisão vs. recall num detector", com a curva aparecendo. Conceito que muita gente confunde.
- **Seção nova no artigo do blog:** entra depois da validação manual (contagem) — a contagem valida o *pipeline* de ponta a ponta; a curva P-R valida o *detector* isoladamente. Os dois níveis de avaliação juntos são o que fecha o argumento "eu entendo e sei medir o que fiz".
- **Portfólio/mestrado:** `mAP@0.5 = X` no seu vídeo, com a metodologia acima, é uma linha de resultado de verdade. Some ao estudo de falha e você tem material de artigo curto para WVC / SIBGRAPI workshop / ENIAC.

## Limitações honestas (diga no conteúdo)

- **mAP@0.5 é generoso.** O padrão COCO reporta a média de mAP para IoU de 0,5 a 0,95. Rodar `--iou 0.75` mostra quão apertadas são realmente as suas caixas — vale reportar os dois.
- **Poucos frames = estimativa ruidosa.** 100–200 frames dão ordem de grandeza confiável, não um número de três casas decimais. Não sobreinterprete a terceira casa.
- **Anotação humana também erra.** Se você não tem certeza de uma caixa (veículo muito ocluído), essa incerteza entra no seu "padrão-ouro". Anote com critério consistente.
