# Correção do vazamento treino/validação

O primeiro split deste dataset foi feito sorteando frames aleatoriamente. Como os
60 frames vêm de um único vídeo amostrado uniformemente, frames vizinhos mostram os
mesmos veículos — e o sorteio colocou o mesmo veículo dos dois lados da divisão. O
`mAP@0,5 = 0,660` publicado media o quanto o modelo lembrava, não o quanto ele
generalizava.

Este documento registra o diagnóstico medido, a correção aplicada e o quanto o
número caiu quando o vazamento foi removido.

## 1. O diagnóstico, em números

Três medições nos rótulos existentes:

- **58% das caixas têm IoU ≥ 0,5 com uma caixa da mesma classe no frame anterior**
  (40% têm IoU ≥ 0,7). É o mesmo veículo, praticamente na mesma posição.
- **Os 12 frames de validação tinham distância 1 para o treino. Todos os doze.**
  Não havia um único frame de validação limpo — não era um split parcialmente
  contaminado, era um split integralmente contaminado.
- **As 28 caixas de "ônibus" são 2 ônibus.** Os frames 7 a 34 têm exatamente um
  ônibus cada, ininterruptamente (27 caixas), mais um no frame 47. O README
  estimava "cinco ou seis"; são dois. E há **1 única caixa de moto** no dataset
  inteiro.

Esse último ponto condena o número que o projeto publicava. O `mAP@0,5 = 0,660` era
a média de três coisas: `ônibus AP = 1,000` (um ônibus parado, decorado), `carro
AP = 0,974` (inflado por vazamento) e `moto AP = 0,000` (uma instância, nunca
detectada). Nenhuma das três parcelas mede generalização, e a média delas não
significa nada.

## 2. O que NÃO era o erro

Vale ser preciso, porque isso determina o remédio: **as anotações estão corretas.**
Um ônibus marcado em 27 frames são 27 caixas certas. Não havia nada para limpar nos
`.txt`, e re-anotar não teria corrigido coisa alguma.

O erro estava em dois outros lugares — **quais frames foram amostrados** (uniformemente
no tempo, sem considerar que veículos persistem entre amostras) e **como treino e
validação foram divididos** (sorteio aleatório sobre uma sequência temporal). O
segundo é o grave, e custa zero para consertar: não exige anotar mais nada.

## 3. A correção

`src/preparar_split.py` monta a divisão por **bloco de tempo contíguo**, com uma
**zona-tampão** de frames descartados na fronteira, para que nenhum veículo apareça
dos dois lados:

```
frames 0–47      frames 48–51     frames 52–65
   TREINO          TAMPÃO          VALIDAÇÃO
  47 frames      4 descartados      9 frames
```

```bash
python src/preparar_split.py --val-inicio 52 --tampao 4
```

A distância mínima entre um frame de validação e um de treino passou de **1 frame
para 5 frames** (~35 s de vídeo) — tempo de sobra para qualquer veículo em movimento
sair de cena.

## 4. O experimento controlado

Para que a diferença seja atribuível ao split e não ao treino, **os dois braços
foram retreinados do zero com hiperparâmetros idênticos** aos do treino original
(`yolov8n.pt`, 60 épocas, batch 8, imgsz 640, seed 0, patience 20). A única variável
que muda é a divisão dos dados.

Controle de sanidade: o braço aleatório retreinado reproduz o modelo histórico —
`mAP@0,5 = 0,652` contra os `0,660` do `best.pt` original (a diferença vem de
treinar em GPU em vez de CPU). O braço é fiel ao que estava publicado.

## 5. O resultado

Comparação honesta é **só na classe carro**: é a única com exemplos nos dois lados
do corte temporal.

| | split aleatório (vazado) | split temporal (limpo) | queda |
|---|---|---|---|
| carro AP@0,5 *(métrica do projeto)* | 0,974 | **0,853** | −0,121 |
| carro AP@0,5 *(métrica do Ultralytics)* | 0,960 | **0,881** | −0,079 |
| carro AP@0,5:0,95 | 0,502 | **0,402** | −0,100 |
| melhor F1 | 0,919 | **0,825** | −0,094 |
| precisão / recall | 0,94 / 0,89 | 0,82 / 0,82 | |

**O vazamento valia cerca de 0,08–0,12 de AP em carro.** Na faixa estrita de IoU
(0,5:0,95) a queda é de 20% relativos, o que faz sentido: o modelo não estava só
reconhecendo que havia um carro, estava reproduzindo a caixa que tinha decorado.

O número honesto do detector nesta cena é **AP ≈ 0,85 em carro**, não `mAP 0,660` —
e os dois números nem medem a mesma coisa.

## 6. O que ficou impossível de avaliar

- **Ônibus.** Com 2 ônibus no dataset, não existe corte que treine e valide a classe
  honestamente: ou o ônibus fica todo no treino, ou aparece nos dois lados. O bloco
  de validação temporal não tem nenhum. O `AP = 1,000` que o projeto publicava era
  um ônibus parado sendo reencontrado, e deve ser retirado como resultado.
- **Moto.** Uma instância no dataset inteiro, com `AP = 0,000` — o modelo nunca a
  detecta. Uma caixa não sustenta nem treino nem avaliação.

Isso não é limitação do split; é limitação dos dados. Duas das três classes do
`data.yaml` não têm exemplos suficientes para existirem.

## 7. Ressalvas do novo número

- **A validação limpa tem 9 frames e 40 caixas, ≈ 17 carros distintos.** É honesta,
  mas é pequena: o −0,12 tem margem de erro larga. A afirmação sustentável é
  qualitativa — *o vazamento inflava o resultado de forma mensurável* — não o valor
  exato da queda.
- **Ainda é uma única gravação, uma única câmera, um único ângulo.** O corte temporal
  remove o vazamento entre treino e validação, mas não torna o resultado válido para
  outra cena. Para responder "o modelo generaliza?", o caminho é validar na segunda
  gravação disponível em `data/raw/`, que o modelo nunca viu.
- **O split temporal introduz um viés próprio:** o fim do vídeo pode ter iluminação
  ou volume de tráfego diferentes do começo (o bloco de validação tem 40 caixas de
  carro em 9 frames, densidade maior que a média do treino). É um preço menor que o
  vazamento, mas existe.
