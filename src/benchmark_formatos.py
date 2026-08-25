"""Mede o trade-off velocidade x acuracia x tamanho entre formatos de deploy.

Formatos comparados, na ordem em que o modelo "desce" para producao:

    PyTorch FP32  ->  ONNX FP32  ->  TensorRT FP16  ->  TensorRT INT8

Para cada um mede, no MESMO conjunto de imagens e com o MESMO imgsz:

  - tamanho do artefato em disco (MB)
  - latencia de inferencia pura (ms/imagem, mediana e p95)
  - latencia fim-a-fim, incluindo pre e pos-processamento (ms/imagem)
  - mAP@IoU e melhor F1, calculados com o nucleo de avaliacao do proprio
    projeto (src/nucleo_avaliacao.py), nao com a metrica interna do YOLO

A separacao entre "inferencia pura" e "fim-a-fim" importa: pre e pos-processamento
rodam em CPU e sao identicos nos quatro formatos, entao diluem o ganho da
quantizacao. Reportar so o numero fim-a-fim esconde o efeito que o estudo mede.

Uso:
    python src/benchmark_formatos.py \
        --modelo runs/detect/runs/treino_trafego/weights/best.pt --classes custom
"""

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from deteccao import detectar, ler_gts, listar_imagens, mapas_de_classe
from nucleo_avaliacao import avaliar_classe, casar_por_imagem, metricas_por_limiar

# Cada formato: (rotulo na tabela, sufixo do arquivo, kwargs do export do ultralytics).
# quantize=16/8 e a API do ultralytics >= 8.4: as flags half=True/int8=True antigas
# nao existem mais, e no TensorRT 11 a precisao e assada no ONNX (ver nota no README).
# bytes_por_peso: precisao em que os pesos ficam guardados. Serve para separar o
# tamanho DOS PESOS (que a quantizacao encolhe) do tamanho DO ARTEFATO em disco
# (que, no caso do .engine, e dominado por codigo de kernel compilado e nao encolhe).
FORMATOS = {
    "pytorch_fp32": ("PyTorch FP32", ".pt", None, 4),
    "onnx_fp32": ("ONNX FP32", ".onnx", {"format": "onnx"}, 4),
    "trt_fp16": ("TensorRT FP16", ".fp16.engine", {"format": "engine", "quantize": 16}, 2),
    "trt_int8": ("TensorRT INT8", ".int8.engine", {"format": "engine", "quantize": 8}, 1),
}
ORDEM_PADRAO = ["pytorch_fp32", "onnx_fp32", "trt_fp16", "trt_int8"]


def tamanho_mb(caminho):
    return Path(caminho).stat().st_size / (1024 * 1024)


def ambiente():
    """Coleta o que faz o numero ser reproduzivel (ou explica por que nao e)."""
    import torch
    import ultralytics

    info = {
        "data_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "so": f"{platform.system()} {platform.release()}",
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda_torch": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }
    try:
        import tensorrt
        info["tensorrt"] = tensorrt.__version__
    except ImportError:
        info["tensorrt"] = None
    try:
        import onnxruntime
        info["onnxruntime"] = onnxruntime.__version__
        info["onnxruntime_providers"] = onnxruntime.get_available_providers()
    except ImportError:
        info["onnxruntime"] = None
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
    if shutil.which("nvidia-smi"):
        info["driver"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True).stdout.strip().splitlines()[0]
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True)
    info["commit"] = commit.stdout.strip() or None
    return info


def tabela_markdown(df, colunas, titulos):
    """Tabela markdown sem depender de tabulate."""
    linhas = [[titulos[c] for c in colunas]]
    for _, r in df.iterrows():
        linhas.append([f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in colunas])
    larg = [max(len(l[i]) for l in linhas) for i in range(len(colunas))]
    def fmt(vals, sep=" | "):
        return "| " + sep.join(v.ljust(w) for v, w in zip(vals, larg)) + " |"
    return "\n".join([fmt(linhas[0]),
                      "|" + "|".join("-" * (w + 2) for w in larg) + "|",
                      *(fmt(l) for l in linhas[1:])])


def obter_artefato(modelo_pt, chave, imgsz, data_calib, refazer):
    """Devolve o caminho do artefato do formato, exportando se ainda nao existir."""
    _, sufixo, kw, _bytes = FORMATOS[chave]
    if kw is None:
        return Path(modelo_pt)

    destino = Path(str(Path(modelo_pt).with_suffix("")) + sufixo)
    if destino.exists() and not refazer:
        print(f"  [{chave}] reaproveitando {destino.name}")
        return destino

    from ultralytics import YOLO
    print(f"  [{chave}] exportando... (engine TensorRT pode levar minutos)")
    kwargs = dict(kw, imgsz=imgsz, device=0)
    if kwargs.get("quantize") == 8:
        # Calibracao INT8 com as imagens do proprio dataset: mesma cena, mesma
        # camera. Calibrar com imagens de outra distribuicao e a causa mais comum
        # de INT8 "perder acuracia" sem que a culpa seja da quantizacao.
        kwargs["data"] = str(data_calib)
        # split="train" NAO e detalhe: o ultralytics calibra no split "val" por
        # padrao (exporter.py, data[self.args.split or "val"]). Como a acuracia
        # tambem e medida em val, o default faria o motor INT8 ver as imagens de
        # avaliacao durante a calibracao. E vazamento mais fraco que treinar em val
        # -- a calibracao le so faixas de ativacao, nao rotulos -- mas continua
        # sendo vazamento, e a linha INT8 da tabela ficaria com vantagem indevida.
        kwargs["split"] = "train"
    saida = Path(YOLO(str(modelo_pt)).export(**kwargs))
    if saida.resolve() != destino.resolve():
        shutil.move(str(saida), str(destino))
    return destino


def medir_latencia(modelo, imagem, imgsz, repeticoes, aquecimento):
    """Latencia por imagem, com sincronizacao de GPU e descarte do aquecimento.

    Usa uma unica imagem ja carregada em memoria para que leitura de disco e
    decodificacao JPEG nao entrem na conta e nao variem entre formatos.
    """
    import torch

    for _ in range(aquecimento):
        modelo.predict(imagem, imgsz=imgsz, verbose=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    e2e, puro = [], []
    for _ in range(repeticoes):
        t0 = time.perf_counter()
        r = modelo.predict(imagem, imgsz=imgsz, verbose=False)[0]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        e2e.append((time.perf_counter() - t0) * 1000)
        puro.append(r.speed["inference"])
    return np.array(e2e), np.array(puro)


def medir_acuracia(modelo, imagens, labels_dir, imgsz, conf_min, iou, classes, mapa_gt):
    """mAP e melhor F1 usando o nucleo de avaliacao do projeto.

    Deliberadamente NAO usa o model.val() do Ultralytics: assim o numero de cada
    formato e comparavel com o resto da documentacao do projeto, que sai todo do
    mesmo nucleo. Duas implementacoes de mAP nao sao a mesma medida.
    """
    preds_por_img, tamanhos = detectar(modelo, imagens, classes, conf_min, imgsz)
    gts_por_img = ler_gts(tamanhos, labels_dir, classes, mapa_gt)

    aps, marcas_todas, n_gt_todas = {}, [], 0
    for c, nome in classes.items():
        res = avaliar_classe(preds_por_img[c], gts_por_img[c], iou)
        if res["n_gt"] > 0:
            aps[nome] = res["ap"]
        for preds, gts in zip(preds_por_img[c], gts_por_img[c]):
            marcas, n_gt = casar_por_imagem(preds, gts, iou)
            marcas_todas.extend(marcas)
            n_gt_todas += n_gt

    mAP = float(np.mean(list(aps.values()))) if aps else 0.0
    limiares = np.round(np.arange(0.05, 0.96, 0.05), 2)
    tabela = pd.DataFrame(metricas_por_limiar(marcas_todas, n_gt_todas, limiares))
    melhor = tabela.loc[tabela["f1"].idxmax()]
    return mAP, aps, float(melhor["f1"]), float(melhor["limiar"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelo", default="models/split_temporal.pt")
    ap.add_argument("--imagens", default="data/splits/temporal/images/val")
    ap.add_argument("--labels", default="data/splits/temporal/labels/val")
    ap.add_argument("--data-calib", default="data/splits/temporal/data.yaml",
                    help="data.yaml com as imagens usadas para calibrar o INT8")
    ap.add_argument("--classes", choices=["coco", "custom"], default="custom")
    ap.add_argument("--formatos", nargs="+", default=ORDEM_PADRAO, choices=ORDEM_PADRAO)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--repeticoes", type=int, default=200)
    ap.add_argument("--aquecimento", type=int, default=30)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--conf-min", type=float, default=0.001)
    ap.add_argument("--refazer-export", action="store_true")
    ap.add_argument("--saida", default="outputs/benchmark_formatos")
    args = ap.parse_args()

    import cv2
    from ultralytics import YOLO

    CLASSES, mapa_gt = mapas_de_classe(args.classes)
    imagens = listar_imagens(args.imagens)
    imagem_latencia = cv2.imread(str(imagens[0]))
    n_params = sum(p.numel() for p in YOLO(args.modelo).model.parameters())

    env = ambiente()
    print(f"Benchmark de formatos | modelo: {args.modelo}")
    print(f"  GPU {env.get('gpu', 'CPU')} | driver {env.get('driver', '?')} | "
          f"torch {env['torch']} | TensorRT {env['tensorrt']}")
    print(f"  {len(imagens)} imagens | imgsz {args.imgsz} | {args.repeticoes} repeticoes "
          f"(+{args.aquecimento} de aquecimento)\n")

    linhas = []
    base = None
    for chave in args.formatos:
        rotulo, _, _, bytes_peso = FORMATOS[chave]
        print(f"{rotulo}")
        try:
            artefato = obter_artefato(args.modelo, chave, args.imgsz,
                                      args.data_calib, args.refazer_export)
            modelo = YOLO(str(artefato), task="detect")
            e2e, puro = medir_latencia(modelo, imagem_latencia, args.imgsz,
                                       args.repeticoes, args.aquecimento)
            mAP, aps, f1, limiar = medir_acuracia(modelo, imagens, args.labels, args.imgsz,
                                                  args.conf_min, args.iou, CLASSES, mapa_gt)
        except Exception as e:
            print(f"  FALHOU: {type(e).__name__}: {e}\n")
            linhas.append({"formato": rotulo, "chave": chave, "erro": f"{type(e).__name__}: {e}"})
            continue

        linha = {
            "formato": rotulo,
            "chave": chave,
            "artefato": str(artefato),
            "tamanho_mb": round(tamanho_mb(artefato), 2),
            "ms_inferencia_mediana": round(float(np.median(puro)), 3),
            "ms_inferencia_p95": round(float(np.percentile(puro, 95)), 3),
            "ms_e2e_mediana": round(float(np.median(e2e)), 3),
            "fps_inferencia": round(1000 / float(np.median(puro)), 1),
            "fps_e2e": round(1000 / float(np.median(e2e)), 1),
            f"mAP@{args.iou}": round(mAP, 4),
            "melhor_f1": round(f1, 4),
            "limiar_melhor_f1": limiar,
            "pesos_mb": round(n_params * bytes_peso / 1e6, 2),
            "erro": "",
        }
        for nome, v in aps.items():
            linha[f"ap_{nome}"] = round(v, 4)
        if base is None:
            base = linha
        linha["speedup_vs_pytorch"] = round(
            base["ms_inferencia_mediana"] / linha["ms_inferencia_mediana"], 2)
        linha["delta_mAP_vs_pytorch"] = round(linha[f"mAP@{args.iou}"] - base[f"mAP@{args.iou}"], 4)
        # Reducao dos PESOS (efeito real da quantizacao). O tamanho em disco do
        # .engine nao acompanha: ver a nota no rodape da tabela.
        linha["reducao_pesos"] = round(base["pesos_mb"] / linha["pesos_mb"], 2)
        linhas.append(linha)

        print(f"  {linha['tamanho_mb']:6.2f} MB | "
              f"{linha['ms_inferencia_mediana']:6.2f} ms inferencia "
              f"({linha['fps_inferencia']:5.1f} FPS) | "
              f"{linha['ms_e2e_mediana']:6.2f} ms e2e | "
              f"mAP@{args.iou} {mAP:.3f} | F1 {f1:.3f}\n")

    df = pd.DataFrame(linhas)
    saida = Path(args.saida)
    saida.parent.mkdir(parents=True, exist_ok=True)
    sufixo = "_" + Path(args.modelo).stem
    csv = saida.with_name(saida.name + sufixo + ".csv")
    js = saida.with_name(saida.name + sufixo + "_ambiente.json")
    df.to_csv(csv, index=False)
    js.write_text(json.dumps({"ambiente": env, "args": vars(args)}, indent=2, default=str))

    ok = df[df["erro"] == ""] if "erro" in df else df
    md = saida.with_name(saida.name + sufixo + ".md")
    if len(ok):
        cols = ["formato", "tamanho_mb", "ms_inferencia_mediana", "fps_inferencia",
                "pesos_mb", f"mAP@{args.iou}", "melhor_f1", "speedup_vs_pytorch",
                "delta_mAP_vs_pytorch", "reducao_pesos"]
        titulos = {"formato": "formato", "tamanho_mb": "tamanho (MB)",
                   "ms_inferencia_mediana": "inferencia (ms)", "fps_inferencia": "FPS",
                   f"mAP@{args.iou}": f"mAP@{args.iou}", "melhor_f1": "melhor F1",
                   "pesos_mb": "pesos (MB)", "speedup_vs_pytorch": "speedup",
                   "delta_mAP_vs_pytorch": "d mAP", "reducao_pesos": "reducao pesos"}
        tabela = tabela_markdown(ok, cols, titulos)
        print("\n" + tabela)
        cabecalho = (
            f"# Trade-off velocidade x acuracia x tamanho - `{Path(args.modelo).name}`\n\n"
            f"- GPU: {env.get('gpu', 'CPU')} | driver {env.get('driver', '?')}\n"
            f"- torch {env['torch']} | ultralytics {env['ultralytics']} | "
            f"TensorRT {env['tensorrt']} | onnxruntime {env['onnxruntime']}\n"
            f"- {len(imagens)} imagens de `{args.imagens}` | imgsz {args.imgsz} | "
            f"batch 1 | {args.repeticoes} repeticoes (+{args.aquecimento} aquecimento)\n"
            f"- IoU {args.iou} | conf minima {args.conf_min} | commit {env['commit']} | "
            f"{env['data_utc']}\n\n"
            "Latencia de inferencia pura (sem pre/pos-processamento), mediana sobre uma\n"
            "unica imagem ja em memoria. Reproduzir com:\n\n"
            f"```bash\npython src/benchmark_formatos.py --modelo {args.modelo} "
            f"--classes {args.classes} \\\n    --imagens {args.imagens} "
            f"--labels {args.labels}\n```\n\n")
        rodape = (
            f"\n\nModelo com {n_params:,} parametros.\n\n".replace(",", ".") +
            "- **pesos (MB)** = parametros x bytes por peso do formato. E o que a\n"
            "  quantizacao encolhe de fato.\n"
            "- **tamanho (MB)** = artefato em disco. Para `.engine` do TensorRT 11 esse\n"
            "  numero e dominado por codigo de kernel compilado (o build reporta\n"
            "  `Total Weights Memory` ~17 bytes por parametro, que nao sao pesos), entao\n"
            "  o engine em disco NAO encolhe com a quantizacao -- chega a crescer, porque\n"
            "  o INT8 adiciona nos de quantiza/dequantiza. Comparar `.engine` com `.pt`\n"
            "  compara coisas diferentes: `.pt` e so peso, `.engine` e peso + kernels.\n"
            "- **speedup** e calculado sobre a inferencia pura, nao o fim-a-fim: pre e\n"
            "  pos-processamento rodam em CPU, sao iguais nos quatro formatos e diluem\n"
            "  o ganho da quantizacao.\n")
        md.write_text(cabecalho + tabela + rodape)
    falhas = df[df["erro"] != ""] if "erro" in df else df.iloc[:0]
    for _, r in falhas.iterrows():
        print(f"\nFALHOU {r['formato']}: {r['erro']}")
    print(f"\nCSV: {csv}\nTabela: {md}\nAmbiente: {js}")


if __name__ == "__main__":
    main()
