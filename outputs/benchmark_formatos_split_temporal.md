# Trade-off velocidade x acuracia x tamanho - `best.pt`

- GPU: NVIDIA GeForce RTX 3050 Laptop GPU | driver 535.230.02
- torch 2.6.0+cu124 | ultralytics 8.4.104 | TensorRT 11.2.1.2 | onnxruntime 1.24.4
- 9 imagens de `data/yolo_dataset_temporal/images/val` | imgsz 640 | batch 1 | 200 repeticoes (+30 aquecimento)
- IoU 0.5 | conf minima 0.001 | commit a13489a | 2026-08-25T00:03:53+00:00

Latencia de inferencia pura (sem pre/pos-processamento), mediana sobre uma
unica imagem ja em memoria. Reproduzir com:

```bash
python src/benchmark_formatos.py --modelo runs/detect/runs/split_estudo/temporal/weights/best.pt --classes custom \
    --imagens data/yolo_dataset_temporal/images/val --labels data/yolo_dataset_temporal/labels/val
```

| formato       | tamanho (MB) | inferencia (ms) | FPS     | pesos (MB) | mAP@0.5 | melhor F1 | speedup | d mAP  | reducao pesos |
|---------------|--------------|-----------------|---------|------------|---------|-----------|---------|--------|---------------|
| PyTorch FP32  | 5.960        | 2.610           | 383.200 | 12.050     | 0.853   | 0.825     | 1.000   | 0.000  | 1.000         |
| ONNX FP32     | 11.700       | 5.048           | 198.100 | 12.050     | 0.870   | 0.815     | 0.520   | 0.017  | 1.000         |
| TensorRT FP16 | 48.070       | 1.425           | 701.600 | 6.020      | 0.858   | 0.805     | 1.830   | 0.005  | 2.000         |
| TensorRT INT8 | 52.580       | 1.506           | 664.000 | 3.010      | 0.849   | 0.783     | 1.730   | -0.004 | 4.000         |

Modelo com 3.011.433 parametros.

- **pesos (MB)** = parametros x bytes por peso do formato. E o que a
  quantizacao encolhe de fato.
- **tamanho (MB)** = artefato em disco. Para `.engine` do TensorRT 11 esse
  numero e dominado por codigo de kernel compilado (o build reporta
  `Total Weights Memory` ~17 bytes por parametro, que nao sao pesos), entao
  o engine em disco NAO encolhe com a quantizacao -- chega a crescer, porque
  o INT8 adiciona nos de quantiza/dequantiza. Comparar `.engine` com `.pt`
  compara coisas diferentes: `.pt` e so peso, `.engine` e peso + kernels.
- **speedup** e calculado sobre a inferencia pura, nao o fim-a-fim: pre e
  pos-processamento rodam em CPU, sao iguais nos quatro formatos e diluem
  o ganho da quantizacao.
