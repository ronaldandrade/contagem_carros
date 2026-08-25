# Velocidade x acurácia x tamanho nos formatos de deploy

O modelo treinado foi levado pela cadeia de formatos que normalmente se usa para
colocar um detector em produção:

```
PyTorch FP32  →  ONNX FP32  →  TensorRT FP16  →  TensorRT INT8
```

A pergunta é simples de enunciar e fácil de responder errado: **quanto se ganha de
velocidade, e quanto se paga em acurácia e em tamanho?**

Tudo é gerado por `src/benchmark_formatos.py`. As tabelas com os números, o
ambiente exato (GPU, driver, versões, commit) e o CSV bruto ficam em `outputs/`.

## Como reproduzir

```bash
# modelo treinado neste dataset (3 classes)
python src/benchmark_formatos.py \
    --modelo models/treino_original.pt --classes custom

# modelo genérico do COCO
python src/benchmark_formatos.py --modelo yolov8n.pt --classes coco
```

Os exports ficam em cache ao lado do `.pt`; use `--refazer-export` para forçar. Os
engines do TensorRT levam ~2,5 min cada para construir e **não são portáveis**:
valem para a GPU, o driver e a versão de TensorRT onde foram gerados.

## Os números

RTX 3050 Laptop (4 GB), driver 535.230.02, TensorRT 11.2.1.2, torch 2.6.0+cu124,
imgsz 640, batch 1, 200 repetições, IoU 0,5.

### Modelo do split temporal (o número válido)

`models/split_temporal.pt`, avaliado nos 9 frames de
validação limpa. É a única tabela cuja coluna de acurácia vale como valor absoluto —
as demais foram medidas no split com vazamento.

| formato       | tamanho (MB) | inferência (ms) | FPS   | pesos (MB) | mAP@0,5 | melhor F1 | speedup |
|---------------|--------------|-----------------|-------|------------|---------|-----------|---------|
| PyTorch FP32  | 5,96         | 2,61            | 383,2 | 12,05      | 0,853   | 0,825     | 1,00    |
| ONNX FP32     | 11,70        | 5,05            | 198,1 | 12,05      | 0,870   | 0,815     | 0,52    |
| TensorRT FP16 | 48,07        | **1,43**        | 701,6 | 6,02       | 0,858   | 0,805     | **1,83**|
| TensorRT INT8 | 52,58        | 1,51            | 664,0 | 3,01       | 0,849   | 0,783     | 1,73    |

Controle: o `mAP@0,5 = 0,853` do PyTorch bate exatamente com o AP em carro medido
pela métrica do projeto em `docs/split_temporal.md`.

### Os outros dois modelos (só o eixo de velocidade)

Modelo do split aleatório (`best.pt`) e modelo genérico (`yolov8n.pt`). A coluna de
acurácia destes foi medida no split com vazamento e **não vale como valor absoluto**;
estão aqui para mostrar que o comportamento dos formatos não depende do modelo.

| modelo | ONNX FP32 | TensorRT FP16 | TensorRT INT8 |
|---|---|---|---|
| split temporal | 0,52× | **1,83×** | 1,73× |
| split aleatório | 0,52× | **1,85×** | 1,75× |
| yolov8n genérico | 0,52× | **1,80×** | 1,77× |

Três modelos, mesmo padrão. **O TensorRT FP16 é o ponto ótimo, com ~1,8× de ganho
sobre o PyTorch e acurácia intacta.** O ONNX no `CUDAExecutionProvider` é a surpresa
desagradável — fica cerca de **duas vezes mais lento que o PyTorch** nos três casos.
Não é erro de medição: verifiquei que ele está mesmo na GPU. O ganho do ONNX Runtime
viria do `TensorrtExecutionProvider`, e aí a comparação seria com o TensorRT, não com
o ONNX puro.

### Cuidado ao ler a coluna de acurácia

No modelo do split temporal os deltas de mAP vão de −0,004 a **+0,017**, com o ONNX
FP32 aparecendo *acima* do PyTorch. ONNX FP32 é numericamente quase idêntico ao
PyTorch: esse +0,017 é ruído da validação de 9 frames e 40 caixas, não ganho.

O `melhor F1` deixa isso explícito ao inverter de sinal conforme o conjunto: no
modelo vazado ele *subia* nos formatos exportados (0,921 → 0,947), no modelo limpo
ele *cai* (0,825 → 0,783). Uma métrica que troca de direção ao trocar o conjunto
está medindo o conjunto. F1 depende de escolher um limiar numa grade de 20 pontos
sobre 40 caixas — instável por construção.

**O que se sustenta:** nenhum formato degradou de forma detectável, e o FP16 entrega
1,8× de graça. **O que não se sustenta:** ler os deltas de acurácia individualmente.

## As quatro decisões de medição que mudam o resultado

Um benchmark de quantização é fácil de fazer de um jeito que não mede nada. Três
escolhas aqui foram deliberadas.

**1. Latência de inferência pura, separada do fim-a-fim.** Pré e pós-processamento
(carregar, redimensionar, NMS, montar as caixas) rodam em CPU e são idênticos nos
quatro formatos. Se o número reportado incluir isso, o ganho da quantização
aparece diluído por um custo fixo que a quantização nunca ia tocar. A tabela traz
as duas colunas, e o `speedup` é calculado sobre a inferência pura.

**2. A latência é medida sobre uma imagem já em memória.** Leitura de disco e
decodificação de JPEG variam com o cache do sistema operacional e não têm nada a
ver com o formato do modelo. São 30 execuções de aquecimento descartadas (a
primeira inferência sempre paga alocação e autotuning) e 200 medidas, com
`torch.cuda.synchronize()` a cada uma — sem isso o relógio para antes da GPU
terminar e todo formato parece infinitamente rápido.

**3. A calibração INT8 usa o split de treino, não o de validação.** O Ultralytics
calibra no split `val` por padrão (`exporter.py`: `data[self.args.split or "val"]`).
Como a acurácia também é medida em `val`, o default faria o motor INT8 observar as
imagens de avaliação durante a calibração. É vazamento mais fraco que treinar em
`val` — a calibração lê faixas de ativação, não rótulos — mas é vazamento, e daria à
linha INT8 uma vantagem indevida. `benchmark_formatos.py` força `split="train"`.

**4. A acurácia usa o `nucleo_avaliacao.py` do projeto, não a métrica interna do
YOLO.** Assim o número de cada formato é comparável com o resto da documentação —
e, de fato, a linha do PyTorch reproduz exatamente o `mAP@0,5 = 0,660` já
registrado em `docs/avaliacao.md`. Se eu usasse o `model.val()` do Ultralytics,
estaria comparando duas implementações de mAP diferentes.

## Tamanho: onde a comparação óbvia engana

A comparação natural seria "tamanho do arquivo do `.pt` contra tamanho do arquivo
do `.engine`". Ela dá um resultado absurdo: o engine FP16 tem **48 MB** contra
**5,96 MB** do `.pt` — a quantização teria deixado o modelo *oito vezes maior*.

Não deixou. O modelo tem 3.011.433 parâmetros, ou seja 12 MB em FP32, 6 MB em
FP16 e 3 MB em INT8, e os ONNX intermediários confirmam isso na balança: 12,27 MB,
6,21 MB e 6,50 MB. O que acontece é que **`.pt` e `.engine` não são a mesma
espécie de arquivo**. O `.pt` guarda pesos. O `.engine` do TensorRT 11 guarda
pesos *mais* o código de kernel compilado e o plano de execução — o próprio log do
build entrega o jogo ao reportar `Total Weights Memory: 51 MB`, que daria 17 bytes
por parâmetro, um número impossível para pesos de verdade.

Por isso a tabela traz duas colunas separadas: `pesos (MB)`, que é o que a
quantização de fato encolhe, e `tamanho (MB)`, que é o arquivo que você realmente
precisa distribuir. Medir a memória de GPU também não ajuda a separar as duas
coisas — FP16 e INT8 ocupam os mesmos 178 MiB no processo, porque o contexto CUDA
domina o consumo num modelo deste tamanho.

A conclusão que interessa: **num modelo de 3 M de parâmetros, quantizar não reduz
o que você distribui nem o que você aloca na GPU.** Reduz os pesos, que já eram a
menor parte do custo. Quem quiser tamanho de artefato menor está olhando para a
alavanca errada.

## Por que o INT8 não ganha do FP16

O INT8 sai ligeiramente *mais lento* que o FP16 e com um engine *maior*. A primeira
suspeita seria quantização que não pegou, mas o grafo desmente: o `best.int8.onnx`
tem **492 tensores de peso em INT8 e 246 pares Q/DQ**, cobrindo as 64 convoluções.
A quantização pegou.

O que acontece é que esses 246 pares de nós de quantiza/dequantiza custam tempo, e
num modelo de 3 M de parâmetros a 640×640 numa RTX 3050 a inferência é dominada
por lançamento de kernel, não por vazão aritmética. Não há trabalho de multiplicação
suficiente para os tensor cores INT8 amortizarem o overhead que eles próprios
introduzem. E os nós Q/DQ extras são também o motivo de o engine INT8 ser maior
que o FP16.

O INT8 compensa em modelos maiores, em batches maiores, ou em hardware onde a
diferença de vazão entre INT8 e FP16 é grande. Nenhuma dessas três condições vale
aqui.

## Ressalvas

- **A acurácia é medida em 12 imagens de validação.** Isso é pouco para afirmar
  qualquer coisa sobre o delta de acurácia entre formatos: o `d mAP` de -0,001 que
  aparece na tabela está bem dentro do ruído dessa amostra, e não deve ser lido
  como "a quantização custou 0,001 de mAP". No modelo treinado, que é o que
  interessa, **nenhum formato degradou de forma visível** — os quatro ficam em
  0,659/0,660. Já no modelo genérico o INT8 cai de 0,028 para 0,013; parece
  drástico em proporção, mas são frações de ponto num modelo cujo mAP nesta cena
  já é 0,02, ou seja, ruído sobre ruído. Some-se a isso o aviso do próprio
  TensorRT durante o build: **a calibração INT8 usou 12 imagens, e o recomendado
  são mais de 300.** Para levar INT8 a sério aqui, calibrar com um conjunto maior
  seria o primeiro passo. O mesmo vazamento treino/validação discutido em `docs/avaliacao.md`
  vale aqui e infla os quatro números por igual — como o interesse é a *diferença*
  entre formatos, o viés se cancela em boa parte.
- **Os números de latência valem para esta GPU.** Uma RTX 3050 Laptop tem pouco
  poder de cálculo em relação ao overhead fixo, o que é exatamente o motivo de o
  INT8 não render. Em uma GPU maior a ordem das linhas pode mudar.
- **`melhor F1` sobe de 0,921 (PyTorch) para 0,947 nos três formatos exportados.**
  Os três exportados concordam entre si e discordam do PyTorch, o que aponta para
  uma diferença no pós-processamento do caminho exportado, não para ganho de
  qualidade. O `mAP`, que não depende da escolha de limiar, fica praticamente
  igual nos quatro — é o número em que confiar para comparar formatos.
